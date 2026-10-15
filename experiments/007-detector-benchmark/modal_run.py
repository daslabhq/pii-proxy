#!/usr/bin/env python3
"""Run detectors on Modal: one container per (model, dataset) job, in parallel.

Same code as run.py (detectors.py, mapping.py), same data files; outputs land
in out/<dataset>/ locally, exactly where run.py would write them.

    modal run modal_run.py                                   # the full matrix (JOBS)
    modal run modal_run.py --only gliner2-pii:spy            # one job
    modal run modal_run.py --only gliner2-pii:spy --limit 50 # smoke test
    modal run modal_run.py --latency                         # CPU latency table
    modal run modal_run.py --build-langs                     # data/ai4privacy-<lang>.jsonl
    modal run modal_run.py --model-facts                     # parameter counts, checkpoint provenance
    modal run modal_run.py --labels-cost                     # CPU latency at 5 / 20 / all labels

The exp-005 checkpoint (pii-proxy-ner) is read from the `pii-proxy-models`
volume; Hub weights are cached on `pii-proxy-hf-cache`.
"""

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.16.1", "gliner==0.2.29", "gliner2[local]==2.1.0",
                 "presidio-analyzer", "spacy", "pythainlp==5.4.0", "numpy", "huggingface_hub",
                 "datasets")
    .run_commands("python -m spacy download en_core_web_lg")
    .env({"HF_HOME": "/hf", "PII_PROXY_NER": "/models/checkpoint/checkpoint-31220"})
    .add_local_file(HERE / "detectors.py", "/root/detectors.py")
    .add_local_file(HERE / "mapping.py", "/root/mapping.py")
    .add_local_file(HERE / "latency.py", "/root/latency.py")
    .add_local_file(HERE / "build_data.py", "/root/build_data.py")
    .add_local_dir(HERE / "data", "/root/data", ignore=["*-placeholders.jsonl"])
)

app = modal.App("pii-proxy-007-detector-benchmark", image=image)
models_volume = modal.Volume.from_name("pii-proxy-models")
hf_cache = modal.Volume.from_name("pii-proxy-hf-cache", create_if_missing=True)
VOLUMES = {"/models": models_volume, "/hf": hf_cache}

GLINER_MODELS = ["gliner2-pii", "nvidia-gliner-pii", "pii-proxy-ner", "pii-proxy-24",
                 "gliner-multi-pii", "knowledgator-pii"]
ALL_MODELS = GLINER_MODELS + ["openai-privacy-filter", "presidio"]
MULTILINGUAL = ["gliner2-pii", "gliner-multi-pii", "openai-privacy-filter", "nvidia-gliner-pii",
                "pii-proxy-ner"]
LANGUAGES = ["de", "fr", "it", "es", "nl"]  # ai4privacy's other five languages

# Reproducing GLiNER2-PII's published SPY table: the protocol it appears to
# have used (whole documents, so GLiNER v1 truncates at 384 tokens; the
# privacy filter through the stock pipeline; SPY's 7 category names as labels).
REPRO = ([(m, "spy", "dataset", False, "trunc") for m in
          ["nvidia-gliner-pii", "gliner-multi-pii", "knowledgator-pii", "pii-proxy-ner"]]
         + [(m, "spy", "spy7", False, "trunc") for m in
            ["gliner2-pii", "nvidia-gliner-pii", "gliner-multi-pii", "knowledgator-pii"]]
         + [("gliner2-pii", "spy", "spy7", False, "trunc"), ("knowledgator-pii", "spy", "dataset", False, "trunc")])

# SPY's seven category names as labels, whole documents (windows): a second
# query strategy for every GLiNER model; score.py picks each model's better
# strategy on SPY's dev split.
SPY7_JOBS = [(m, "spy", "spy7", False) for m in GLINER_MODELS]

# (model, dataset, vocab, segment_thai)
JOBS = (
    [(m, d, "dataset", False) for m in ALL_MODELS for d in ["spy", "ai4privacy", "nemotron"]]
    + [(m, d, "full", False) for m in GLINER_MODELS for d in ["spy", "ai4privacy"]]
    + [(m, "nemotron", "full", False) for m in ["nvidia-gliner-pii", "pii-proxy-ner"]]
    + [(m, "thai", v, seg) for m in MULTILINGUAL for v in ["dataset", "full"] for seg in [False, True]
       if not (m == "openai-privacy-filter" and v == "full")]
    + [(m, f"ai4privacy-{lang}", v, False) for lang in LANGUAGES for m in GLINER_MODELS
       for v in ["dataset", "full"]]
    + [("openai-privacy-filter", f"ai4privacy-{lang}", "dataset", False) for lang in LANGUAGES]
    + [("gliner2-pii-char", "thai", "dataset", False)]  # GLiNER2's splitter for unspaced scripts
)


@app.function(gpu="A10G", volumes=VOLUMES, timeout=60 * 60 * 3)
def run_job(model: str, dataset: str, vocab: str, segment_thai: bool, limit: int = 0,
            mode: str = "", batch_size: int = 16) -> dict:
    return _run(model, dataset, vocab, segment_thai, limit, device="cuda", mode=mode,
                batch_size=batch_size)


@app.function(cpu=4, memory=8192, volumes=VOLUMES, timeout=60 * 60 * 3)
def run_job_cpu(model: str, dataset: str, vocab: str, segment_thai: bool, limit: int = 0) -> dict:
    return _run(model, dataset, vocab, segment_thai, limit, device="cpu")


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=60 * 60 * 2)
def latency_one_container(models: list, n: int = 30) -> list:
    """Every model timed in the same container, one after another: Modal's CPU
    containers differ by machine (NVIDIA's model measured 6.9 s in one and
    10.8 s in another), so only a shared machine makes the rows comparable."""
    import gc
    import sys
    sys.path.insert(0, "/root")
    from latency import measure
    rows = []
    for model in models:
        rows.append({**measure(model, n, threads=2), "cpu": "2 CPU cores, one container"})
        gc.collect()
    hf_cache.commit()
    return rows


@app.function(cpu=2, memory=4096, volumes=VOLUMES, timeout=60 * 60)
def latency_server_class(model: str, n: int = 50) -> dict:
    """Per-text latency on a 2-core / 4 GB container, batch size 1."""
    return _latency(model, n)


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=60 * 30)
def build_languages() -> dict:
    """ai4privacy's validation split in its five other languages, drawn on Modal
    so the dataset download stays off the machine running this."""
    import sys
    sys.path.insert(0, "/root")
    import build_data
    out = {code: build_data.build_ai4privacy_language(code) for code in LANGUAGES}
    hf_cache.commit()
    return out


@app.function(cpu=4, memory=16384, volumes=VOLUMES, timeout=60 * 60)
def facts() -> dict:
    """Parameter count of every model as loaded, and what checkpoint-31220 is."""
    import sys
    sys.path.insert(0, "/root")
    import gc
    from detectors import load_runner
    params = {}
    for model in GLINER_MODELS + ["openai-privacy-filter"]:
        runner = load_runner(model, "cpu")
        params[model] = sum(p.numel() for p in runner.model.parameters())
        del runner
        gc.collect()
    ckpt = Path("/models/checkpoint/checkpoint-31220")
    state = json.loads((ckpt / "trainer_state.json").read_text())
    hf_cache.commit()
    return {"params": params,
            "checkpoint_files": sorted(p.name for p in ckpt.iterdir()),
            "checkpoint_dirs": sorted(p.name for p in ckpt.parent.iterdir()),
            "trainer_state": {k: v for k, v in state.items() if k != "log_history"},
            "log_tail": state.get("log_history", [])[-3:]}


@app.function(cpu=2, memory=4096, volumes=VOLUMES, timeout=60 * 60)
def label_sweep(model: str) -> list:
    """Per-text CPU latency at 5, 20 and all labels, in one container."""
    import sys
    sys.path.insert(0, "/root")
    from latency import label_sweep as sweep
    from mapping import query_labels
    full = len(query_labels(model, "spy", "full"))
    rows = sweep(model, counts=(5, 20, full))
    hf_cache.commit()
    return [{**r, "cpu": "2 CPU cores / 4 GB"} for r in rows]


def _run(model, dataset, vocab, segment_thai, limit, device, mode="", batch_size=16):
    import sys
    sys.path.insert(0, "/root")
    from detectors import FLOOR, HF_IDS, detect, load_runner
    from mapping import query_labels

    records = [json.loads(l) for l in open(f"/root/data/{dataset}.jsonl")]
    records = records[:limit] if limit else records
    fixed = model.startswith("openai-privacy-filter") or model == "presidio"
    labels = [] if fixed else query_labels(model, dataset, vocab)
    runner = load_runner(model, device)
    hf_cache.commit()
    t0 = time.perf_counter()
    found = detect(runner, [r["text"] for r in records], labels, segment_thai=segment_thai,
                   batch_size=batch_size, windows=mode != "trunc")
    seconds = time.perf_counter() - t0
    lines = [json.dumps({"id": r["id"], "detections": d}, ensure_ascii=False)
             for r, d in zip(records, found)]
    meta = {"model": model, "hf_id": HF_IDS[model], "dataset": dataset, "vocab": vocab,
            "segment_thai": segment_thai, "mode": mode or "windows", "labels": labels,
            "threshold_floor": FLOOR,
            "device": f"modal {device}", "records": len(records), "seconds": round(seconds, 1),
            "ms_per_record": round(1000 * seconds / len(records), 1), "versions": versions()}
    return {"jsonl": "\n".join(lines) + "\n", "meta": meta}


def versions():
    from importlib.metadata import version
    return {p: version(p) for p in ["torch", "transformers", "gliner", "gliner2",
                                    "presidio-analyzer", "spacy", "pythainlp"]}


def _latency(model, n):
    import sys
    sys.path.insert(0, "/root")
    from latency import measure
    result = measure(model, n, threads=2)
    hf_cache.commit()
    return {**result, "cpu": "2 CPU cores / 4 GB"}


@app.local_entrypoint()
def main(only: str = "", limit: int = 0, latency: bool = False, skip_done: bool = True,
         repro: bool = False, build_langs: bool = False, model_facts: bool = False,
         labels_cost: bool = False):
    if labels_cost:
        rows = [r for rs in label_sweep.map(["pii-proxy-ner", "nvidia-gliner-pii", "gliner2-pii"]) for r in rs]
        (HERE / "latency-labels.json").write_text(json.dumps(rows, indent=2))
        for r in rows:
            print(r)
        return
    if build_langs:
        from build_data import write
        for code, records in build_languages.remote().items():
            write(f"ai4privacy-{code}", records)
        return
    if model_facts:
        result = facts.remote()
        (HERE / "facts.json").write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str)[:3000])
        return
    if latency:
        rows = latency_one_container.remote(ALL_MODELS)
        (HERE / "latency-server-class.json").write_text(json.dumps(rows, indent=2, default=str))
        for r in rows:
            print(r)
        return
    jobs = [j for j in JOBS + SPY7_JOBS if not only or f"{j[0]}:{j[1]}" in only.split(",")]
    if only and not jobs:  # a job outside the matrix, e.g. --only gliner2-pii:spy:full
        m, d, *rest = only.split(":")
        jobs = [(m, d, rest[0] if rest else "dataset", "seg" in rest)]
    if repro:
        return run_repro(limit)
    if skip_done and not limit:
        jobs = [j for j in jobs if not output_path(*j).exists()]
    print(f"{len(jobs)} jobs")
    gpu_jobs = [j for j in jobs if j[0] != "presidio"]
    cpu_jobs = [j for j in jobs if j[0] == "presidio"]
    calls = ([(j, run_job.spawn(*j, limit)) for j in gpu_jobs]
             + [(j, run_job_cpu.spawn(*j, limit)) for j in cpu_jobs])
    for job, call in calls:
        try:
            result = call.get()
        except Exception as e:  # keep the other jobs' outputs
            print(f"FAILED {job}: {type(e).__name__}: {str(e)[:300]}")
            continue
        path = output_path(*job, limited=bool(limit))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result["jsonl"])
        path.with_suffix(".meta.json").write_text(json.dumps(result["meta"], indent=2, ensure_ascii=False))
        print(f"{job}: {result['meta']['records']} records, {result['meta']['ms_per_record']} ms/record -> {path}")


def run_repro(limit):
    todo = [j for j in REPRO if limit or not repro_path(*j).exists()]
    calls = [(j, run_job.spawn(*j[:4], limit, j[4], 2)) for j in todo]
    for (model, dataset, vocab, _, mode), call in calls:
        try:
            result = call.get()
        except Exception as e:
            print(f"FAILED {model} {vocab} {mode}: {type(e).__name__}: {str(e)[:300]}")
            continue
        path = repro_path(model, dataset, vocab, False, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result["jsonl"])
        path.with_suffix(".meta.json").write_text(json.dumps(result["meta"], indent=2))
        print(f"{path.stem}: {result['meta']['records']} records -> {path}")


def repro_path(model, dataset, vocab, _seg, mode) -> Path:
    name = model + (f".{vocab}" if vocab != "dataset" else "") + (f".{mode}" if mode else "")
    return HERE / "out-repro" / dataset / f"{name}.jsonl"


def output_path(model, dataset, vocab, segment_thai, limited=False) -> Path:
    name = model + (f".{vocab}" if vocab != "dataset" else "") + (".seg" if segment_thai else "")
    return HERE / ("out-smoke" if limited else "out") / dataset / f"{name}.jsonl"
