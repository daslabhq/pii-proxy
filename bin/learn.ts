#!/usr/bin/env bun
/**
 * pii-proxy learn — Tag traces/texts with PII entity labels using an LLM.
 *
 * Reads input texts (plain text files, JSONL traces, or Nemotron-PII format),
 * runs PII detection (regex + LLM), and outputs labeled spans ready for
 * fine-tuning a compiled NER model.
 *
 * Usage:
 *   # Tag plain text files
 *   bun run bin/learn.ts --input ./notes/*.txt --out ./compiled/labels.jsonl
 *
 *   # Tag OTel JSONL traces (extracts text from span attributes)
 *   bun run bin/learn.ts --traces ./traces/*.jsonl --out ./compiled/labels.jsonl
 *
 *   # Use a specific model
 *   bun run bin/learn.ts --input ./notes/*.txt --model qwen3:4b --out ./compiled/labels.jsonl
 *
 *   # Use a cloud API
 *   bun run bin/learn.ts --input ./notes/*.txt \
 *     --endpoint https://api.anthropic.com/v1/messages \
 *     --model claude-sonnet-4-20250514 \
 *     --out ./compiled/labels.jsonl
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { dirname, resolve } from 'path';
import { learn, type LearnOptions } from '../src/learn.js';

// ─── Args ───────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts: Record<string, string | string[]> = {};
  let i = 0;
  while (i < args.length) {
    if (args[i].startsWith('--')) {
      const key = args[i].slice(2);
      const values: string[] = [];
      i++;
      while (i < args.length && !args[i].startsWith('--')) {
        values.push(args[i]);
        i++;
      }
      opts[key] = values.length === 1 ? values[0] : values;
    } else {
      i++;
    }
  }
  return opts;
}

// ─── Input loaders ──────────────────────────────────────────────

function loadPlainTexts(paths: string[]): string[] {
  const texts: string[] = [];
  for (const p of paths) {
    const resolved = resolve(p);
    if (!existsSync(resolved)) {
      console.error(`File not found: ${resolved}`);
      continue;
    }
    texts.push(readFileSync(resolved, 'utf8').trim());
  }
  return texts;
}

function loadOtelTraces(paths: string[]): string[] {
  const texts: string[] = [];
  for (const p of paths) {
    const resolved = resolve(p);
    if (!existsSync(resolved)) continue;
    const lines = readFileSync(resolved, 'utf8').trim().split('\n');
    for (const line of lines) {
      try {
        const span = JSON.parse(line);
        const attrs = span.attributes ?? {};

        // Extract text from common attribute patterns
        const candidates = [
          attrs['tool.input'],
          attrs['tool.output'],
          attrs['llm.request.body'],
          attrs['scene.value'],
          ...(Object.entries(attrs)
            .filter(([k]) => k.includes('message.content'))
            .map(([, v]) => v)),
        ].filter(Boolean);

        for (const c of candidates) {
          const text = typeof c === 'string' ? c : JSON.stringify(c);
          if (text.length > 50) texts.push(text);
        }
      } catch {
        // skip malformed lines
      }
    }
  }
  return texts;
}

function loadJsonlRecords(paths: string[]): string[] {
  const texts: string[] = [];
  for (const p of paths) {
    const resolved = resolve(p);
    if (!existsSync(resolved)) continue;
    const lines = readFileSync(resolved, 'utf8').trim().split('\n');
    for (const line of lines) {
      try {
        const record = JSON.parse(line);
        if (typeof record.text === 'string') {
          texts.push(record.text);
        }
      } catch {
        // skip
      }
    }
  }
  return texts;
}

// ─── Main ───────────────────────────────────────────────────────

const opts = parseArgs();

if (!opts.input && !opts.traces && !opts.jsonl) {
  console.log(`pii-proxy learn — Tag texts with PII labels using an LLM

Usage:
  bun run bin/learn.ts --input ./notes/*.txt --out ./labels.jsonl
  bun run bin/learn.ts --traces ./traces/*.jsonl --out ./labels.jsonl
  bun run bin/learn.ts --jsonl ./data.jsonl --out ./labels.jsonl

Options:
  --input <files>       Plain text files to tag
  --traces <files>      OTel JSONL trace files (extracts text from span attributes)
  --jsonl <files>       JSONL with {"text": "..."} records
  --model <name>        LLM model (default: qwen3:1.7b)
  --endpoint <url>      LLM API endpoint (default: http://localhost:11434/v1/chat/completions)
  --out <path>          Output JSONL path (default: ./compiled/labels.jsonl)
  --limit <n>           Max records to process
`);
  process.exit(0);
}

// Load texts
let texts: string[] = [];
if (opts.input) {
  const paths = Array.isArray(opts.input) ? opts.input : [opts.input];
  texts = loadPlainTexts(paths);
  console.log(`Loaded ${texts.length} text files`);
} else if (opts.traces) {
  const paths = Array.isArray(opts.traces) ? opts.traces : [opts.traces];
  texts = loadOtelTraces(paths);
  console.log(`Extracted ${texts.length} texts from OTel traces`);
} else if (opts.jsonl) {
  const paths = Array.isArray(opts.jsonl) ? opts.jsonl : [opts.jsonl];
  texts = loadJsonlRecords(paths);
  console.log(`Loaded ${texts.length} records from JSONL`);
}

if (texts.length === 0) {
  console.error('No texts found. Check your input paths.');
  process.exit(1);
}

// Apply limit
const limit = opts.limit ? parseInt(opts.limit as string, 10) : texts.length;
texts = texts.slice(0, limit);
console.log(`Processing ${texts.length} records\n`);

// Configure
const learnOpts: LearnOptions = {
  llm: {
    model: (opts.model as string) ?? undefined,
    endpoint: (opts.endpoint as string) ?? undefined,
  },
  onProgress(done, total, record) {
    const pct = Math.round((done / total) * 100);
    const spans = record.spans.length;
    const preview = record.text.slice(0, 60).replace(/\n/g, ' ');
    console.log(`[${done}/${total}] ${pct}% — ${spans} entities — "${preview}..."`);
  },
};

// Run
const startTime = Date.now();
const result = await learn(texts, learnOpts);
const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

// Output
const outPath = resolve((opts.out as string) ?? './compiled/labels.jsonl');
const outDir = dirname(outPath);
if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });

const lines = result.records.map(r => JSON.stringify(r));
writeFileSync(outPath, lines.join('\n') + '\n');

// Stats
console.log(`\n${'═'.repeat(60)}`);
console.log(`Learn complete in ${elapsed}s`);
console.log(`${'═'.repeat(60)}`);
console.log(`  Records tagged:  ${result.stats.total_records}`);
console.log(`  Total spans:     ${result.stats.total_spans}`);
console.log(`  Failed:          ${result.stats.failed_records}`);
console.log(`\n  Entity distribution:`);
const sorted = Object.entries(result.stats.entity_counts).sort((a, b) => b[1] - a[1]);
for (const [type, count] of sorted) {
  console.log(`    ${type.padEnd(20)} ${count}`);
}
console.log(`\n  Output: ${outPath}`);
console.log(`\n  Next step: fine-tune a NER model with:`);
console.log(`    python3 experiments/finetune_ner.py --labels ${outPath}`);
