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
Diagnosis: Type 2 diabetes, hypertension
Medication: Metformin 1000mg 2x/day, Lisinopril 10mg 1x/day
Insurance: TK 109876543
Contact: marcus.weber@gmail.com, +49 170 1234567
Address: Hauptstraße 42, 68161 Mannheim, Germany
Notes: Patient reports improved diet compliance. HbA1c dropped from 8.1 to 7.2.
Referred by Dr. Anika Hoffmann, Hausarztpraxis Mannheim.
Follow-up scheduled in 3 months.`;

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
      content: `You are a clinical AI assistant. Analyze this patient record and provide:
1. A brief clinical summary
2. Risk assessment
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
