#!/usr/bin/env bun
/**
 * Evaluate pii-proxy's LLM tagger against Nemotron-PII ground truth.
 *
 * Downloads healthcare records from NVIDIA's Nemotron-PII dataset,
 * runs our LLM tagger on the raw text, and compares our detected
 * spans against their gold labels.
 *
 * Prerequisites: ollama serve + qwen3:1.7b
 *
 * Usage:
 *   bun run experiments/eval_tagger.ts
 *   bun run experiments/eval_tagger.ts --model qwen3:4b --limit 50
 */
import { learn } from '../src/learn.js';

// ─── Config ─────────────────────────────────────────────────────

const args = process.argv.slice(2);
const model = args.includes('--model') ? args[args.indexOf('--model') + 1] : 'qwen3:1.7b';
const limit = args.includes('--limit') ? parseInt(args[args.indexOf('--limit') + 1]) : 20;

// Map Nemotron labels to our coarser types (same mapping as finetune_ner.py)
const LABEL_MAP: Record<string, string> = {
  first_name: 'person_name',
  last_name: 'person_name',
  middle_name: 'person_name',
  medical_record_number: 'medical_record',
  date_of_birth: 'date_of_birth',
  date: 'date',
  date_time: 'date',
  time: 'date',
  email: 'email',
  phone_number: 'phone',
  street_address: 'location',
  city: 'location',
  state: 'location',
  county: 'location',
  zip_code: 'location',
  country: 'location',
  ssn: 'national_id',
  health_plan_beneficiary_number: 'insurance_id',
  certificate_license_number: 'national_id',
  biometric_identifier: 'national_id',
  url: 'url',
  unique_id: 'national_id',
  blood_type: 'medical_info',
  pin: 'national_id',
  password: 'credential',
  swift_bic: 'financial_id',
};

// ─── Load Nemotron-PII ──────────────────────────────────────────

console.log('╔══════════════════════════════════════════════════╗');
console.log('║  Tagger Eval — pii-proxy vs Nemotron-PII        ║');
console.log('╚══════════════════════════════════════════════════╝');
console.log();
console.log(`Model: ${model}`);
console.log(`Limit: ${limit} records`);
console.log();

console.log('Loading Nemotron-PII healthcare records...');

// Download via Python (handles HuggingFace datasets properly)
const cacheFile = 'experiments/output/nemotron_healthcare_cache.jsonl';

interface NemotronSpan {
  start: number;
  end: number;
  text: string;
  label: string;
}

interface GoldRecord {
  text: string;
  goldSpans: Array<{ start: number; end: number; text: string; label: string }>;
}

const records: GoldRecord[] = [];

// Check cache first
const cacheExists = await Bun.file(cacheFile).exists();
if (cacheExists) {
  const lines = (await Bun.file(cacheFile).text()).trim().split('\n');
  for (const line of lines) {
    const row = JSON.parse(line);
    records.push(row);
    if (records.length >= limit) break;
  }
  console.log(`Loaded ${records.length} records from cache`);
} else {
  // Fetch via Python
  console.log('Fetching from HuggingFace (first time, will cache)...');
  const pyScript = `
import ast, json, sys
from datasets import load_dataset

ds = load_dataset('nvidia/Nemotron-PII', split='train', streaming=True)
label_map = ${JSON.stringify(LABEL_MAP)}

records = []
total = 0
for row in ds:
    total += 1
    domain = row.get('domain', '') or ''
    if 'health' not in domain.lower():
        if total >= 20000: break
        continue
    spans_raw = ast.literal_eval(row['spans']) if isinstance(row['spans'], str) else row['spans']
    gold = []
    for s in spans_raw:
        mapped = label_map.get(s['label'])
        if mapped:
            gold.append({'start': s['start'], 'end': s['end'], 'text': s['text'], 'label': mapped})
    if gold:
        records.append({'text': row['text'], 'goldSpans': gold})
    if total >= 20000: break

for r in records:
    print(json.dumps(r))
sys.stderr.write(f'Exported {len(records)} healthcare records\\n')
`;

  const proc = Bun.spawn(['python3', '-c', pyScript], { stdout: 'pipe', stderr: 'pipe' });
  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  if (stderr) console.log(`  ${stderr.trim()}`);

  // Cache and load
  await Bun.write(cacheFile, stdout);
  const lines = stdout.trim().split('\n');
  for (const line of lines) {
    records.push(JSON.parse(line));
    if (records.length >= limit) break;
  }
}

console.log(`Loaded ${records.length} healthcare records with ground truth\n`);

// ─── Run our tagger ─────────────────────────────────────────────

console.log(`Running pii-proxy learn (${model})...\n`);

const texts = records.map(r => r.text);
const startTime = Date.now();

const result = await learn(texts, {
  llm: { model },
  onProgress(done, total, record) {
    const pct = Math.round((done / total) * 100);
    console.log(`  [${done}/${total}] ${pct}% — ${record.spans.length} entities`);
  },
});

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`\nTagging done in ${elapsed}s (${(parseFloat(elapsed) / records.length).toFixed(1)}s/record)\n`);

// ─── Compare against ground truth ───────────────────────────────

interface MatchStats {
  tp: number;  // our span matches a gold span (type + overlap)
  fp: number;  // we detected something that's not in gold
  fn: number;  // gold has a span we missed
}

const perType: Record<string, MatchStats> = {};
const overall: MatchStats = { tp: 0, fp: 0, fn: 0 };

function getStats(type: string): MatchStats {
  if (!perType[type]) perType[type] = { tp: 0, fp: 0, fn: 0 };
  return perType[type];
}

function spansOverlap(a: { start: number; end: number }, b: { start: number; end: number }): boolean {
  return a.start < b.end && b.start < a.end;
}

for (let i = 0; i < records.length; i++) {
  const gold = records[i].goldSpans;
  const pred = result.records[i]?.spans ?? [];

  const goldMatched = new Set<number>();
  const predMatched = new Set<number>();

  // Match: same type + overlapping positions
  for (let pi = 0; pi < pred.length; pi++) {
    for (let gi = 0; gi < gold.length; gi++) {
      if (goldMatched.has(gi)) continue;
      if (pred[pi].label === gold[gi].label && spansOverlap(pred[pi], gold[gi])) {
        predMatched.add(pi);
        goldMatched.add(gi);
        overall.tp++;
        getStats(pred[pi].label).tp++;
        break;
      }
    }
  }

  // False positives: we detected, not in gold
  for (let pi = 0; pi < pred.length; pi++) {
    if (!predMatched.has(pi)) {
      overall.fp++;
      getStats(pred[pi].label).fp++;
    }
  }

  // False negatives: in gold, we missed
  for (let gi = 0; gi < gold.length; gi++) {
    if (!goldMatched.has(gi)) {
      overall.fn++;
      getStats(gold[gi].label).fn++;
    }
  }
}

// ─── Report ─────────────────────────────────────────────────────

function f1(s: MatchStats) {
  const precision = s.tp / (s.tp + s.fp) || 0;
  const recall = s.tp / (s.tp + s.fn) || 0;
  const f1 = 2 * precision * recall / (precision + recall) || 0;
  return { precision, recall, f1 };
}

console.log('═'.repeat(72));
console.log('Tagger Evaluation Results');
console.log('═'.repeat(72));
console.log();
console.log(`${'Entity'.padEnd(20)} ${'Prec'.padStart(8)} ${'Recall'.padStart(8)} ${'F1'.padStart(8)} ${'TP'.padStart(6)} ${'FP'.padStart(6)} ${'FN'.padStart(6)}`);
console.log('─'.repeat(72));

const sortedTypes = Object.entries(perType).sort((a, b) => {
  const aTotal = a[1].tp + a[1].fn;
  const bTotal = b[1].tp + b[1].fn;
  return bTotal - aTotal;
});

for (const [type, stats] of sortedTypes) {
  const m = f1(stats);
  console.log(
    `${type.padEnd(20)} ${(m.precision * 100).toFixed(1).padStart(7)}% ${(m.recall * 100).toFixed(1).padStart(7)}% ${(m.f1 * 100).toFixed(1).padStart(7)}% ${String(stats.tp).padStart(6)} ${String(stats.fp).padStart(6)} ${String(stats.fn).padStart(6)}`
  );
}

console.log('─'.repeat(72));
const m = f1(overall);
console.log(
  `${'OVERALL'.padEnd(20)} ${(m.precision * 100).toFixed(1).padStart(7)}% ${(m.recall * 100).toFixed(1).padStart(7)}% ${(m.f1 * 100).toFixed(1).padStart(7)}% ${String(overall.tp).padStart(6)} ${String(overall.fp).padStart(6)} ${String(overall.fn).padStart(6)}`
);

console.log();
console.log(`Model: ${model}`);
console.log(`Records: ${records.length}`);
console.log(`Time: ${elapsed}s (${(parseFloat(elapsed) / records.length).toFixed(1)}s/record)`);
console.log();
console.log('This measures how well our LLM tagger labels data for fine-tuning.');
console.log('These labels become training data → compiled NER model (9ms, 97.9% F1).');

// Save labeled output for fine-tuning
const outPath = 'experiments/output/tagger_labels.jsonl';
const jsonl = result.records.map(r => JSON.stringify(r)).join('\n');
await Bun.write(outPath, jsonl + '\n');
console.log(`\nLabeled data saved to: ${outPath}`);
