#!/usr/bin/env bun
/**
 * Example: Using pii-proxy with the Anthropic SDK.
 *
 * Masks PII before sending to Claude, then unmasks the response
 * so you can write real values back to your database.
 *
 * Usage:
 *   export ANTHROPIC_API_KEY=sk-...
 *   bun run examples/anthropic-agent.ts
 */
import Anthropic from '@anthropic-ai/sdk';
import { PrivacyProxy } from '../src/index.js';

const proxy = new PrivacyProxy();
const client = new Anthropic();

// ─── Real user data (what your agent receives) ─────────────────

const userEmail = {
  from: 'alex@example.com',
  to: 'support@shop.co.th',
  subject: 'Where is my order?',
  body: `Hi, I placed an order last week and haven't received it yet.
My tracking number is AETH0000345323DY.
Please send updates to alex@example.com or call +49 170 1234567.
Order was shipped to: Hauptstr. 42, 68161 Mannheim, Germany.`,
};

console.log('── Original ──');
console.log(JSON.stringify(userEmail, null, 2));
console.log();

// ─── Mask before sending to the LLM ────────────────────────────

const { masked, detections } = await proxy.maskObject(userEmail);

console.log('── Masked (what Claude sees) ──');
console.log(JSON.stringify(masked, null, 2));
console.log();
console.log(`Detected ${detections.length} PII entities:`);
for (const d of detections) {
  console.log(`  ${d.type}: "${d.value}" → "${d.replacement}"`);
}
console.log();

// ─── Send to Claude ─────────────────────────────────────────────

const response = await client.messages.create({
  model: 'claude-sonnet-4-20250514',
  max_tokens: 512,
  messages: [
    {
      role: 'user',
      content: `Extract structured data from this customer email. Return JSON with fields: customer_email, tracking_number, phone, shipping_address, issue_summary.

Email:
From: ${masked.from}
To: ${masked.to}
Subject: ${masked.subject}
Body: ${masked.body}`,
    },
  ],
});

const llmText = response.content[0].type === 'text' ? response.content[0].text : '';

console.log('── Claude response (with fake values) ──');
console.log(llmText);
console.log();

// ─── Unmask — real values restored for DB write-back ────────────

const real = proxy.unmask(llmText);

console.log('── Unmasked (ready for your database) ──');
console.log(real);
console.log();
console.log(`Map size: ${proxy.size} entities tracked`);
