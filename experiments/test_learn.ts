#!/usr/bin/env bun
/**
 * End-to-end test: learn tags clinical notes, output is valid training data.
 *
 * Prerequisites: ollama serve + qwen3:1.7b
 *
 * Usage:
 *   bun run experiments/test_learn.ts
 */
import { learn } from '../src/learn.js';

const clinicalNotes = [
  `Patient: Marcus Weber (DOB: 15.03.1987)
MRN: MRN-2024-08391
Provider: Dr. Sarah Chen, Universitätsklinikum Heidelberg
Diagnosis: Type 2 diabetes, hypertension
Insurance: TK 109876543
Contact: marcus.weber@gmail.com, +49 170 1234567
Address: Hauptstraße 42, 68161 Mannheim, Germany
Notes: Patient reports improved diet compliance. HbA1c dropped from 8.1 to 7.2.
Referred by Dr. Anika Hoffmann, Hausarztpraxis Mannheim.`,

  `Patient: Elena Petrov (DOB: 22.07.1964)
MRN: MRN-2024-11205
Provider: Dr. James Liu, Stanford Medical Center
Diagnosis: Stage 2 breast cancer, BRCA1+
Insurance: Blue Cross 8847291
Contact: elena.petrov@outlook.com, +1 650 555 0142
Address: 1847 El Camino Real, Palo Alto, CA 94301
Notes: Post-chemo follow-up. Tumor markers declining. Lives with husband Viktor.
Scheduled for MRI at Lucile Packard next month.`,

  `Patient: Ayumi Tanaka (DOB: 03.11.1992)
MRN: MRN-2024-33087
Provider: Dr. Kenta Sato, Tokyo University Hospital
Diagnosis: Generalized anxiety disorder, insomnia
Insurance: National Health 442891037
Contact: a.tanaka@gmail.com, +81 90 1234 5678
Address: 3-chome-14, Bunkyo-ku, Tokyo 113-0033
Notes: Started on sertraline 50mg. Reports high work stress (consultant at McKinsey Tokyo).
Referred by Dr. Yuki Yamamoto, Shinjuku Mental Health Clinic.`,

  `Patient: Omar Al-Rashid (DOB: 08.05.1978)
MRN: MRN-2024-20156
Provider: Dr. Lisa Bergström, Karolinska University Hospital
Diagnosis: Chronic kidney disease stage 3, gout
Insurance: Försäkringskassan 19780508-4321
Contact: omar.alrashid@yahoo.se, +46 70 123 4567
Address: Vasagatan 15, 111 20 Stockholm, Sweden
Notes: eGFR stable at 42. Wife Fatima assists with medication adherence.
Referred by Dr. Erik Johansson, Södermalm Vårdcentral.`,

  `Patient: Maria Silva (DOB: 14.09.2001)
MRN: MRN-2024-45230
Provider: Dr. Carlos Mendes, Hospital São Paulo
Diagnosis: Type 1 diabetes, celiac disease
Insurance: SUS 298.456.123-90
Contact: maria.silva@uol.com.br, +55 11 98765 4321
Address: Rua Augusta 1200, São Paulo, SP 01304-001
Notes: Pump therapy working well. A1c at 6.8. Works as a teacher at Escola Paulista.
Father João accompanies appointments. Next visit in 8 weeks.`,
];

console.log('╔══════════════════════════════════════════════════╗');
console.log('║  pii-proxy learn — End-to-End Test              ║');
console.log('║  Tagging clinical notes with LLM + regex        ║');
console.log('╚══════════════════════════════════════════════════╝');
console.log();
console.log(`Input: ${clinicalNotes.length} clinical notes`);
console.log(`Model: qwen3:1.7b (local via Ollama)`);
console.log();

const startTime = Date.now();

const result = await learn(clinicalNotes, {
  llm: { model: 'qwen3:1.7b' },
  onProgress(done, total, record) {
    const pct = Math.round((done / total) * 100);
    console.log(`[${done}/${total}] ${pct}% — ${record.spans.length} entities found`);
  },
});

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

console.log(`\n${'═'.repeat(60)}`);
console.log(`Done in ${elapsed}s`);
console.log(`${'═'.repeat(60)}`);
console.log(`  Records:  ${result.stats.total_records}`);
console.log(`  Spans:    ${result.stats.total_spans}`);
console.log(`  Failed:   ${result.stats.failed_records}`);
console.log(`\n  Entity distribution:`);
const sorted = Object.entries(result.stats.entity_counts).sort((a, b) => b[1] - a[1]);
for (const [type, count] of sorted) {
  console.log(`    ${type.padEnd(20)} ${count}`);
}

// Show detailed output for first record
console.log(`\n${'─'.repeat(60)}`);
console.log('Sample output (Record 1):');
console.log(`${'─'.repeat(60)}`);
const r = result.records[0];
console.log(`Text: "${r.text.slice(0, 100)}..."\n`);
console.log('Spans:');
for (const s of r.spans) {
  const verified = r.text.slice(s.start, s.end) === s.text;
  const check = verified ? '✓' : '✗';
  console.log(`  [${s.start}:${s.end}] ${s.label.padEnd(18)} "${s.text}" ${check}`);
}

// Verify all spans have correct positions
let totalSpans = 0;
let validSpans = 0;
for (const rec of result.records) {
  for (const s of rec.spans) {
    totalSpans++;
    if (rec.text.slice(s.start, s.end) === s.text) validSpans++;
  }
}
console.log(`\nSpan position accuracy: ${validSpans}/${totalSpans} (${(validSpans/totalSpans*100).toFixed(1)}%)`);

// Output as training data JSONL
const jsonl = result.records.map(r => JSON.stringify(r)).join('\n');
const outPath = 'experiments/output/learn_test_labels.jsonl';
await Bun.write(outPath, jsonl + '\n');
console.log(`\nTraining data written to: ${outPath}`);
console.log(`Ready for: python3 experiments/finetune_ner.py --labels ${outPath}`);
