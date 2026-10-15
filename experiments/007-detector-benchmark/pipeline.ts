#!/usr/bin/env bun
/**
 * pii-proxy as it would ship: a model detector in front of the regex layer
 * (and the Thai pack on Thai text), merged by the library's own detectAll —
 * not a re-implementation of it. The model's detections were computed on
 * Modal (modal_run.py); here they are replayed through a Detector, so the
 * overlap resolution is exactly what PrivacyProxy would do with
 *   new PrivacyProxy({ use: [thai()], detectors: [model, ...defaultDetectors] })
 *
 * Writes, per dataset in out/:
 *   regex.jsonl              the regex layer alone (+ Thai pack on thai)
 *   <model>+regex.jsonl      model (full native vocabulary, its default
 *                            threshold) first, then the regex layer
 *
 *   bun experiments/007-detector-benchmark/pipeline.ts
 */
import { readdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { defaultDetectors, detectAll, type Detection, type Detector } from '../../src/detectors/index.js';
import { thai } from '../../src/packs/th/index.js';

const HERE = import.meta.dir;
const DEFAULT_THRESHOLD: Record<string, number> = { 'knowledgator-pii': 0.3, 'openai-privacy-filter': 0, presidio: 0 };

type Stored = { start: number; end: number; type: string; score?: number };

async function main(): Promise<void> {
  const datasets = readdirSync(join(HERE, 'out'), { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);
  for (const dataset of datasets) {
    const records = readJsonl(join(HERE, 'data', `${dataset}.jsonl`));
    const packs = dataset === 'thai' ? thai().detectors ?? [] : [];
    await writeSystem(dataset, 'regex', records, () => [...packs, ...defaultDetectors]);
    for (const { file, model, seg } of modelOutputs(dataset)) {
      const found = new Map(readJsonl(join(HERE, 'out', dataset, file)).map((r) => [r.id, r.detections as Stored[]]));
      const threshold = DEFAULT_THRESHOLD[model] ?? 0.5;
      await writeSystem(dataset, `${model}+regex${seg}`, records,
        (id) => [...packs, replay(found.get(id) ?? [], threshold), ...defaultDetectors]);
    }
  }
}

/** Model outputs queried the way a detector would query them: GLiNER models
 *  with their full vocabulary; fixed-vocabulary models as they are. */
function modelOutputs(dataset: string): Array<{ file: string; model: string; seg: string }> {
  return readdirSync(join(HERE, 'out', dataset)).flatMap((file) => {
    const m = file.match(/^([a-z0-9-]+)\.full(\.seg)?\.jsonl$/)
      ?? file.match(/^(openai-privacy-filter)(\.seg)?\.jsonl$/);
    return m ? [{ file, model: m[1]!, seg: m[2] ?? '' }] : [];
  });
}

function replay(stored: Stored[], threshold: number): Detector {
  return {
    type: 'model',
    detect: (text) => stored
      .filter((d) => (d.score ?? 1) >= threshold)
      .map((d) => ({ type: d.type, value: text.slice(d.start, d.end), start: d.start, end: d.end, score: d.score, source: 'model' }) as Detection),
  };
}

async function writeSystem(dataset: string, name: string, records: any[], detectorsFor: (id: string) => Detector[]) {
  const lines: string[] = [];
  for (const rec of records) {
    const found = await detectAll(rec.text, detectorsFor(rec.id));
    lines.push(JSON.stringify({
      id: rec.id,
      detections: found.map((d: any) => ({ start: d.start, end: d.end, type: d.type, source: d.source ?? 'regex', ...(d.score != null ? { score: d.score } : {}) })),
    }));
  }
  writeFileSync(join(HERE, 'out', dataset, `${name}.jsonl`), lines.join('\n') + '\n');
  console.log(`${dataset}/${name}: ${records.length} records`);
}

function readJsonl(path: string): any[] {
  return readFileSync(path, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
}

await main();
