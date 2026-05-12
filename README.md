# pii-proxy

[![Test](https://github.com/mirkokiefer/pii-proxy/actions/workflows/test.yml/badge.svg)](https://github.com/mirkokiefer/pii-proxy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Privacy proxy for AI agents. Mask PII before sending to LLMs, unmask responses to write back to real systems.

## Why

Your AI agent processes emails, spreadsheets, CRM data. You don't want to send real names, emails, and tracking numbers to Claude or GPT. But token-based masking (`PERSON_1`, `EMAIL_2`) degrades model quality — LLMs reason poorly over meaningless tokens.

**pii-proxy** replaces PII with plausible fake values — the LLM sees realistic data and reasons correctly. A bijective map lets you reverse everything when writing back to your database.

## Install

```bash
npm install pii-proxy
```

## Quick start

```typescript
import { PrivacyProxy } from 'pii-proxy';

const proxy = new PrivacyProxy();

// Mask PII with plausible fakes
const masked = proxy.mask(
  "Ship order to alex@example.com, tracking AETH0000345323DY"
);
// → "Ship order to alex@johnson.net, tracking BFUI0000482918EZ"

// Send masked.text to your LLM...

// Reverse all fakes back to real values
const real = proxy.unmask(llmResponse);
// "I'll notify alex@johnson.net" → "I'll notify alex@example.com"
```

## How it works

1. **Detect** — regex-based detectors find emails, tracking numbers, IPs, UUIDs, credit cards, phone numbers, and URLs with tokens.
2. **Replace** — each entity is replaced with a plausible fake of the same type (an email becomes another email, a tracking number keeps the same format).
3. **Map** — a bijective map ensures the same real value always maps to the same fake, and vice versa. Consistent within a session, reversible at any time.

```
Real:   "Contact alex@example.com about AETH0000345323DY"
         ↓ mask()
Fake:   "Contact alex@johnson.net about BFUI0000482918EZ"
         ↓ send to LLM → get response
LLM:    "I've emailed alex@johnson.net about the shipment"
         ↓ unmask()
Real:   "I've emailed alex@example.com about the shipment"
```

## Entity types

| Type | Detection | Fake replacement |
|---|---|---|
| Email | `user@domain.com` | Realistic fake email |
| Phone | `+1-234-567-8901` | Format-preserving fake |
| Credit card | Luhn-validated numbers | Valid fake card number |
| IP address | IPv4 addresses | Random valid IP |
| UUID | Standard UUID format | Random UUID |
| URL | URLs with query params/tokens | Sanitized URL |
| Tracking number | UPS, USPS, DHL, AliExpress, etc. | Format-preserving fake |

## Structured data

Mask entire objects (e.g., tool call inputs):

```typescript
const { masked } = proxy.maskObject({
  to: "alex@example.com",
  subject: "Order update",
  body: "Tracking: AETH0000345323DY",
  metadata: { ip: "10.0.0.1" }
});

// masked.to → "alex@johnson.net"
// masked.subject → "Order update" (no PII, unchanged)
// masked.body → "Tracking: BFUI0000482918EZ"
// masked.metadata.ip → "172.45.123.89"

// Reverse everything
const original = proxy.unmaskObject(masked);
```

## Persistence

Save and restore the map across sessions:

```typescript
// Save
const data = proxy.getMap().serialize();
await redis.set('pii-session:123', data);

// Restore in a new process
const proxy2 = new PrivacyProxy();
proxy2.loadMap(await redis.get('pii-session:123'));
proxy2.unmask(text); // works with the same mappings
```

## Example: Anthropic SDK integration

Full round-trip — mask user data, send to Claude, unmask the response for your database ([examples/anthropic-agent.ts](examples/anthropic-agent.ts)):

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { PrivacyProxy } from 'pii-proxy';

const proxy = new PrivacyProxy();
const client = new Anthropic();

const userEmail = {
  from: 'alex@example.com',
  body: 'Tracking number is AETH0000345323DY. Call me at +49 170 1234567.',
};

// Mask before sending to Claude
const { masked } = proxy.maskObject(userEmail);
// masked.from → "cornell62@hotmail.com"
// masked.body → "Tracking number is FCRQ6925552830IZ. Call me at +381.714.0024 x10865."

const response = await client.messages.create({
  model: 'claude-sonnet-4-20250514',
  max_tokens: 512,
  messages: [{ role: 'user', content: `Extract data from: ${JSON.stringify(masked)}` }],
});

// Claude responds with fake values → unmask to get real values for DB
const real = proxy.unmask(response.content[0].text);
// "cornell62@hotmail.com" → "alex@example.com"
// "FCRQ6925552830IZ" → "AETH0000345323DY"
```

## Roadmap

- [x] **v0.1** — Regex detection, faker replacement, bijective round-trip
- [ ] **v0.2** — NER-based name/location detection (optional Presidio backend)
- [ ] **v0.3** — Tool-aware selective masking (keep location real for hotel search, mask for email)
- [ ] **v0.4** — Persistent map backends (Redis, SQLite)
- [ ] **v0.5** — Anthropic/OpenAI SDK middleware (drop-in agent integration)

## License

MIT
