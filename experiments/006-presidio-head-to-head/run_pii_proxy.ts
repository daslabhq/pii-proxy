#!/usr/bin/env bun
/**
 * Run pii-proxy's regex detector layer over the eval sample.
 *
 * This is the shipped zero-setup detection layer (defaultDetectors) —
 * no LLM, no GLiNER. The model layer is benchmarked separately in
 * run_gliner.py and unioned at scoring time.
 *
 * Output: out/pii-proxy-regex.jsonl  {id, detections: [{start, end, type}]}
 */
import { detectAll } from '../../src/detectors/index.js';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const here = import.meta.dir;
mkdirSync(join(here, 'out'), { recursive: true });

const lines = readFileSync(join(here, 'data', 'sample.jsonl'), 'utf8')
  .trim().split('\n');

const out: string[] = [];
const times: number[] = [];
for (const line of lines) {
  const rec = JSON.parse(line);
  const t0 = performance.now();
  const detections = await detectAll(rec.text);
  times.push(performance.now() - t0);
  out.push(JSON.stringify({
    id: rec.id,
    detections: detections.map(d => ({ start: d.start, end: d.end, type: d.type })),
  }));
}

writeFileSync(join(here, 'out', 'pii-proxy-regex.jsonl'), out.join('\n') + '\n');
const avg = times.reduce((a, b) => a + b, 0) / times.length;
console.log(`Done: ${lines.length} records, avg ${avg.toFixed(2)}ms -> out/pii-proxy-regex.jsonl`);
