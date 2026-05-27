#!/usr/bin/env bun
/**
 * Health data demo — local LLM detects patient names, organizations,
 * and locations that regex can't catch. Full round-trip with Claude.
 *
 * Prerequisites:
 *   1. Install Ollama: https://ollama.com
 *   2. Pull a model: ollama pull qwen3:1.7b
 *   3. Start Ollama: ollama serve
 *
 * Usage:
 *   export ANTHROPIC_API_KEY=sk-...
 *   bun run examples/health-data.ts
 */
import Anthropic from '@anthropic-ai/sdk';
import { PrivacyProxy } from '../src/index.js';

const proxy = PrivacyProxy.withLocalLlm({ model: 'qwen3:1.7b' });
const client = new Anthropic();

const patientRecord = `Patient: Marcus Weber (DOB: 15.03.1987)
MRN: MRN-2024-08391
Provider: Dr. Sarah Chen, Universitätsklinikum Heidelberg
Diagnosis: Myopia progression, bilateral astigmatism
Rx: OD -3.25 -0.75 x 180, OS -2.75 -1.00 x 175
Insurance: TK 109876543
Contact: marcus.weber@gmail.com, +49 170 1234567
Address: Hauptstraße 42, 68161 Mannheim, Germany
Notes: Patient reports increased screen time (10+ hrs/day, software engineer).
Referred by Dr. Anika Hoffmann, Augenarzt Mannheim.
Follow-up scheduled in 6 months. Consider ortho-k lenses if progression continues.`;

console.log('╔══════════════════════════════════════════════════╗');
console.log('║  pii-proxy — Health Data Privacy Demo           ║');
console.log('║  Local LLM detection + Cloud LLM reasoning      ║');
console.log('╚══════════════════════════════════════════════════╝');
console.log();

console.log('── Patient Record (real data) ──');
console.log(patientRecord);
console.log();

// ─── Mask with regex + local LLM ────────────────────────────────

console.log('Detecting PII (regex + local LLM via Ollama)...');
const { text: maskedText, detections } = await proxy.mask(patientRecord);

console.log();
console.log(`Found ${detections.length} PII entities:`);
for (const d of detections) {
  const tag = d.type.padEnd(16);
  console.log(`  ${tag} "${d.value}" → "${d.replacement}"`);
}

console.log();
console.log('── Masked Record (what Claude sees) ──');
console.log(maskedText);
console.log();

// ─── Send to Claude for analysis ────────────────────────────────

console.log('Sending masked data to Claude for analysis...');
console.log();

const response = await client.messages.create({
  model: 'claude-sonnet-4-20250514',
  max_tokens: 1024,
  messages: [
    {
      role: 'user',
      content: `You are an ophthalmology AI assistant. Analyze this patient record and provide:
1. A brief clinical summary
2. Risk assessment for myopia progression
3. Recommended next steps
4. A draft follow-up letter to the referring physician

Patient Record:
${maskedText}`,
    },
  ],
});

const llmText = response.content[0].type === 'text' ? response.content[0].text : '';

console.log('── Claude Analysis (fake values) ──');
console.log(llmText);
console.log();

// ─── Unmask — real patient data restored ────────────────────────

const real = proxy.unmask(llmText);

console.log('── Final Output (real values restored) ──');
console.log(real);
console.log();
console.log(`PII entities tracked: ${proxy.size}`);
console.log();
console.log('Patient data never left your infrastructure unmasked.');
console.log('Full audit trail available via proxy.getMap().');
