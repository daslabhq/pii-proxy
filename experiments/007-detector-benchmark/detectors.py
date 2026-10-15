"""Model runners for experiment 007. Every model sees the same windows.

    runner = load_runner(model, device)
    detect(runner, texts, labels, ...) -> [[{start, end, type, score}], ...]

A text longer than one window is cut into overlapping word windows; each
window owns the stretch of text between the midpoints of its overlaps, and a
detection is kept only by the window that owns its start. GLiNER-family models
truncate past 384 words and SPY documents run ~500, so without this a model
silently never sees the contact details at the end of a letter.

Detections are stored at a low threshold (FLOOR) with their scores. Greedy
flat decoding keeps the highest-scoring spans first, so dropping everything
below t afterwards gives exactly what a run at threshold t would have given;
one run serves the default threshold and the tuned one.
"""

import os
import re
from dataclasses import dataclass

FLOOR = 0.2
WINDOW_WORDS = 200
OVERLAP_WORDS = 40

HF_IDS = {
    "gliner2-pii": "fastino/gliner2-privacy-filter-PII-multi",
    "gliner2-pii-char": "fastino/gliner2-privacy-filter-PII-multi (word_splitter=char)",
    "nvidia-gliner-pii": "nvidia/gliner-PII",
    # modal_train.py --full --full-labels: gliner_small-v2.1 on Nemotron-PII's 55 labels
    # (checkpoint-31220, the model behind the main README's 91.5% head-to-head)
    "pii-proxy-ner": os.environ.get("PII_PROXY_NER", "model/pii-proxy-ner"),
    "pii-proxy-24": "daslabhq/pii-proxy",
    "gliner-multi-pii": "urchade/gliner_multi_pii-v1",
    "knowledgator-pii": "knowledgator/gliner-pii-base-v1.0",
    "openai-privacy-filter": "openai/privacy-filter",
    "openai-privacy-filter-argmax": "openai/privacy-filter (transformers pipeline, argmax)",
    "presidio": "presidio-analyzer (default recognizers, spaCy en_core_web_lg)",
}

# The threshold each model's own card uses; everything else is tuned on dev.
DEFAULT_THRESHOLD = {"knowledgator-pii": 0.3}


def detect(runner, texts: list, labels: list, segment_thai: bool = False,
           batch_size: int = 8, windows: bool = True) -> list:
    """windows=False hands each whole text to the model, which then truncates
    as its library does (GLiNER v1: 384 tokens) — only to reproduce published
    numbers that were measured that way."""
    prepared = [ThaiSegmented(t) if segment_thai else Plain(t) for t in texts]
    split = split_windows if windows else (lambda t: [Window(t, 0, 0, len(t))])
    windows = [(i, w) for i, p in enumerate(prepared) for w in split(p.text)]
    raw = runner.predict([w.text for _, w in windows], labels, batch_size)
    found = [[] for _ in texts]
    for (i, w), dets in zip(windows, raw):
        for d in dets:
            if w.owns(d["start"]):
                # GLiNER2 reports an end one past the text for a span that runs to its end
                end = min(d["end"], len(w.text))
                found[i].append({**d, "start": d["start"] + w.offset, "end": end + w.offset})
    return [prepared[i].to_original(sorted(f, key=lambda d: d["start"])) for i, f in enumerate(found)]


def load_runner(model: str, device: str):
    if model == "gliner2-pii":
        return Gliner2Runner(HF_IDS[model], device)
    if model == "gliner2-pii-char":
        return Gliner2Runner(HF_IDS["gliner2-pii"], device, word_splitter="char")
    if model == "openai-privacy-filter":
        return PrivacyFilterRunner(HF_IDS[model], device)
    if model == "openai-privacy-filter-argmax":
        return PrivacyFilterPipelineRunner(HF_IDS["openai-privacy-filter"], device)
    if model == "presidio":
        return PresidioRunner()
    return GlinerRunner(HF_IDS[model], device)


# ─── Windows ────────────────────────────────────────────────────

@dataclass
class Window:
    text: str
    offset: int
    own_start: int  # in document coordinates
    own_end: int

    def owns(self, local_start: int) -> bool:
        return self.own_start <= local_start + self.offset < self.own_end


def split_windows(text: str) -> list:
    words = [m.start() for m in re.finditer(r"\S+", text)]
    if len(words) <= WINDOW_WORDS:
        return [Window(text, 0, 0, len(text))]
    stride = WINDOW_WORDS - OVERLAP_WORDS
    starts = list(range(0, len(words) - OVERLAP_WORDS, stride))
    windows = []
    for k, first in enumerate(starts):
        last = min(first + WINDOW_WORDS, len(words))
        begin = words[first]
        end = words[last] if last < len(words) else len(text)
        own_start = 0 if k == 0 else words[first + OVERLAP_WORDS // 2]
        own_end = len(text) if k == len(starts) - 1 else words[starts[k + 1] + OVERLAP_WORDS // 2]
        windows.append(Window(text[begin:end], begin, own_start, own_end))
    return windows


class Plain:
    def __init__(self, text):
        self.text = text

    def to_original(self, dets):
        return dets


class ThaiSegmented:
    """Thai is written without spaces between words, and GLiNER-family models
    can only start and end a span on a token of their own splitter. GLiNER2's
    splitter keeps Thai vowel and tone marks inside a word, so an unspaced run
    is one long token: title, name and particle together. GLiNER v1's `\w`
    splitter breaks at every mark instead, into pieces that cross word
    boundaries ("คุณศิริลักษณ์" splits as ค|ุ|ณศ|ิ|…, the title's last letter
    glued to the name's first). This puts a space between the words
    pythainlp's newmm segmenter finds — only where Thai script is on either
    side, so IDs, phone numbers and emails stay whole — runs the model on
    that, and maps spans back to the original text."""

    THAI = re.compile(r"[\u0e00-\u0e7f]")

    def __init__(self, text):
        from pythainlp.tokenize import word_tokenize
        chars, self.index, cursor, prev = [], [], 0, ""
        for tok in word_tokenize(text, engine="newmm", keep_whitespace=True):
            thai_edge = self.THAI.search(tok[:1]) or self.THAI.search(prev[-1:])
            if chars and thai_edge and not tok.isspace() and not chars[-1].isspace():
                chars.append(" ")
                self.index.append(None)
            for ch in tok:
                chars.append(ch)
                self.index.append(cursor)
                cursor += 1
            prev = tok
        self.text = "".join(chars)

    def to_original(self, dets):
        out = []
        for d in dets:
            inside = [i for i in self.index[d["start"]:d["end"]] if i is not None]
            if inside:
                out.append({**d, "start": inside[0], "end": inside[-1] + 1})
        return out


# ─── Runners ────────────────────────────────────────────────────

class GlinerRunner:
    def __init__(self, path, device):
        from gliner import GLiNER
        local = path.startswith(("/", "model/"))
        self.model = GLiNER.from_pretrained(path, local_files_only=local).to(device)
        self.model.eval()

    def predict(self, texts, labels, batch_size):
        results = self.model.inference(texts, labels, flat_ner=True, threshold=FLOOR,
                                       batch_size=batch_size)
        return [[{"start": e["start"], "end": e["end"], "type": e["label"],
                  "score": round(float(e["score"]), 4)} for e in ents] for ents in results]


class Gliner2Runner:
    """word_splitter="char" is GLiNER2's own splitter for languages written
    without spaces: one token per character outside Latin runs."""

    def __init__(self, path, device, word_splitter=None):
        from gliner2 import GLiNER2
        self.model = GLiNER2.from_pretrained(path)
        if word_splitter:
            self.model.set_word_splitter(word_splitter)
        self.model.to(device)
        self.model.eval()

    def predict(self, texts, labels, batch_size):
        results = self.model.batch_extract_entities(
            texts, labels, batch_size=batch_size, threshold=FLOOR,
            include_confidence=True, include_spans=True)
        out = []
        for r in results:
            dets = []
            for label, ents in r["entities"].items():
                for e in ents:
                    dets.append({"start": e["start"], "end": e["end"], "type": label,
                                 "score": round(float(e["confidence"]), 4)})
            out.append(dets)
        return out


class PrivacyFilterRunner:
    """OpenAI Privacy Filter: per-token BIOES logits, decoded with the constrained
    Viterbi its card describes, at the default operating point (all six
    transition biases 0 in viterbi_calibration.json, so path score = summed
    log-probabilities under the BIOES transition constraints). Labels are
    fixed by the model; the `labels` argument is ignored."""

    def __init__(self, path, device):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path, dtype=torch.float32).to(device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.allowed = bioes_transitions([self.id2label[i] for i in range(len(self.id2label))])

    def predict(self, texts, labels, batch_size):
        out = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._batch(texts[i:i + batch_size]))
        return out

    def _batch(self, texts):
        import numpy as np
        enc = self.tokenizer(texts, return_offsets_mapping=True, return_tensors="pt", padding=True)
        offsets = enc.pop("offset_mapping")
        with self.torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        logprobs = self.torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        results = []
        for b, text in enumerate(texts):
            n = int(enc["attention_mask"][b].sum())
            pad_left = self.tokenizer.padding_side == "left"
            rows = slice(len(logprobs[b]) - n, None) if pad_left else slice(0, n)
            path = viterbi(logprobs[b][rows], *self.allowed)
            labels_seq = [self.id2label[int(k)] for k in path]
            results.append(bioes_spans(text, labels_seq, offsets[b][rows].tolist(),
                                       np.exp(logprobs[b][rows])))
        return results


class PrivacyFilterPipelineRunner:
    """The privacy filter through transformers' stock token-classification
    pipeline (per-token argmax, aggregation "simple") — what its card's
    pipeline example does, without the constrained Viterbi decoder."""

    def __init__(self, path, device):
        from transformers import pipeline
        self.pipe = pipeline("token-classification", model=path, device=device,
                             aggregation_strategy="simple")

    def predict(self, texts, labels, batch_size):
        out = []
        for ents in self.pipe(texts, batch_size=batch_size):
            dets = []
            for e in ents:
                s, t = e["start"], e["end"]
                dets.append({"start": s, "end": t, "type": e["entity_group"].split("-")[-1],
                             "score": round(float(e["score"]), 4)})
            out.append(dets)
        return [[d for d in strip_spans(text, ds)] for text, ds in zip(texts, out)]


def strip_spans(text, dets):
    out = []
    for d in dets:
        s, e = d["start"], d["end"]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append({**d, "start": s, "end": e})
    return out


class PresidioRunner:
    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]})
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())

    def predict(self, texts, labels, batch_size):
        return [[{"start": r.start, "end": r.end, "type": r.entity_type, "score": round(r.score, 4)}
                 for r in self.analyzer.analyze(text=t, language="en")] for t in texts]


# ─── BIOES decoding (privacy filter) ────────────────────────────

def bioes_transitions(labels):
    """Allowed start / transition / end masks for BIOES tags."""
    import numpy as np
    n = len(labels)
    tag = [(l.split("-", 1) + [None])[:2] if l != "O" else ["O", None] for l in labels]
    start = np.array([t[0] in ("O", "B", "S") for t in tag])
    end = np.array([t[0] in ("O", "E", "S") for t in tag])
    trans = np.zeros((n, n), dtype=bool)
    for i, (a, ca) in enumerate(tag):
        for j, (b, cb) in enumerate(tag):
            if a in ("O", "E", "S"):
                trans[i, j] = b in ("O", "B", "S")
            else:  # B or I: continue the same category
                trans[i, j] = b in ("I", "E") and ca == cb
    return start, trans, end


def viterbi(logprobs, start, trans, end):
    import numpy as np
    neg = -1e9
    T, n = logprobs.shape
    score = np.where(start, logprobs[0], neg)
    back = np.zeros((T, n), dtype=int)
    penalty = np.where(trans, 0.0, neg)
    for t in range(1, T):
        cand = score[:, None] + penalty
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logprobs[t]
    score = np.where(end, score, neg)
    path = [int(score.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1]


def bioes_spans(text, labels_seq, offsets, probs):
    spans, open_span = [], None
    for k, (label, (s, e)) in enumerate(zip(labels_seq, offsets)):
        if s == e:  # special tokens
            continue
        kind, cat = (label.split("-", 1) + [None])[:2] if label != "O" else ("O", None)
        if kind in ("B", "S"):
            open_span = {"start": s, "end": e, "type": cat, "p": [probs[k].max()]}
        elif kind in ("I", "E") and open_span:
            open_span["end"] = e
            open_span["p"].append(probs[k].max())
        if kind in ("E", "S") and open_span:
            spans.append(open_span)
            open_span = None
    out = []
    for sp in spans:
        s, e = sp["start"], sp["end"]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append({"start": s, "end": e, "type": sp["type"],
                        "score": round(float(min(sp["p"])), 4)})
    return out
