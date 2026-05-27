# pii-proxy

[![Test](https://github.com/daslabhq/pii-proxy/actions/workflows/test.yml/badge.svg)](https://github.com/daslabhq/pii-proxy/actions/workflows/test.yml)
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
const masked = await proxy.mask(
  "Ship order to alex@example.com, tracking AETH0000345323DY"
);
// → "Ship order to alex@johnson.net, tracking BFUI0000482918EZ"

// Send masked.text to your LLM...

// Reverse all fakes back to real values
const real = proxy.unmask(llmResponse);
// "I'll notify alex@johnson.net" → "I'll notify alex@example.com"
```

## Local LLM detection

Regex catches emails, IPs, tracking numbers. But what about `"Patient: Marcus Weber"`? That's a name — no regex will reliably find it.

**v0.2** adds pluggable detection with a local LLM layer. A model running on your machine (via [Ollama](https://ollama.com)) detects names, organizations, locations, and domain-specific entities. The PII detection itself never leaves your infrastructure.

```typescript
import { PrivacyProxy } from 'pii-proxy';

// Regex detectors + local LLM via Ollama
const proxy = PrivacyProxy.withLocalLlm({ model: 'qwen3:1.7b' });

const masked = await proxy.mask(
  "Patient Marcus Weber, treated at Universitätsklinikum Heidelberg. Contact: marcus.weber@gmail.com"
);
// → "Patient James Thompson, treated at Bradtke Medical. Contact: lizeth53@yahoo.com"
```

Setup:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (qwen3:1.7b is fast and great for entity detection, ~1.4GB)
ollama pull qwen3:1.7b
```

### Layered pipeline

Detection runs in layers — fast regex first, then LLM for what regex can't catch:

```
Text ──→ [Regex Layer] ──→ [Local LLM Layer] ──→ Deduplicated detections
           emails            person names
           phones             organizations
           IPs                locations
           UUIDs              medical records
           credit cards       insurance IDs
           tracking #s        custom entities
```

Overlapping detections are deduplicated automatically (regex wins ties).

### Configuring the LLM detector

```typescript
import { PrivacyProxy, LlmDetector, defaultDetectors } from 'pii-proxy';

// Use a different model
const proxy = PrivacyProxy.withLocalLlm({ model: 'qwen3:0.6b' });

// Point to a remote Ollama instance or any OpenAI-compatible API
const proxy2 = PrivacyProxy.withLocalLlm({
  endpoint: 'https://your-server.com/v1/chat/completions',
  model: 'gpt-4o-mini',
});

// Detect only specific entity types (faster, more focused)
const proxy3 = PrivacyProxy.withLocalLlm({
  entityTypes: ['person_name', 'organization'],
});

// Full control — compose your own detector stack
const proxy4 = new PrivacyProxy({
  detectors: [
    ...defaultDetectors,                          // regex layer
    new LlmDetector({ model: 'qwen3:1.7b' }),    // LLM layer
    // add more layers here — custom NER, dictionary lookup, etc.
  ],
});
```

Detectors run in order. Each returns `Detection[]` (or `Promise<Detection[]>` for async). Overlapping detections are resolved by position — earlier detectors win ties.

## How it works

1. **Detect** — layered pipeline finds PII entities (regex + optional local LLM).
2. **Replace** — each entity is replaced with a plausible fake of the same type (an email becomes another email, a name becomes another name).
3. **Map** — a bijective map ensures the same real value always maps to the same fake, and vice versa. Consistent within a session, reversible at any time.

```
Real:   "Contact Marcus Weber at marcus@example.com"
         ↓ mask()
Fake:   "Contact James Thompson at cornell62@hotmail.com"
         ↓ send to LLM → get response
LLM:    "I've drafted an email to James Thompson"
         ↓ unmask()
Real:   "I've drafted an email to Marcus Weber"
```

## Entity types

| Type | Detection | Fake replacement |
|---|---|---|
| Email | Regex | Realistic fake email |
| Phone | Regex | Format-preserving fake |
| Credit card | Regex + Luhn | Valid fake card number |
| IP address | Regex | Random valid IP |
| UUID | Regex | Random UUID |
| URL | Regex | Sanitized URL |
| Tracking number | Regex (UPS, USPS, DHL, etc.) | Format-preserving fake |
| Person name | Local LLM | Faker name |
| Organization | Local LLM | Faker company |
| Location | Local LLM | Faker address/city |
| Date of birth | Local LLM | Format-preserving fake date |
| Medical record | Local LLM | Format-preserving fake |
| Insurance ID | Local LLM | Format-preserving fake |
| *Custom* | Local LLM | Format-preserving fallback |

## Structured data

Mask entire objects (e.g., tool call inputs):

```typescript
const { masked } = await proxy.maskObject({
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

## Custom detectors

Any object with a `detect(text)` method is a detector. Use this to add domain-specific patterns, call external NER APIs, or integrate your own models:

```typescript
import { PrivacyProxy, defaultDetectors, LlmDetector } from 'pii-proxy';

// Domain-specific: detect German health insurance numbers (Versichertennummer)
const germanInsuranceDetector = {
  detect(text) {
    const re = /\b[A-Z]\d{9}\b/g;
    const results = [];
    let m;
    while ((m = re.exec(text)) !== null) {
      results.push({ type: 'insurance_id', value: m[0], start: m.index, end: m.index + m[0].length });
    }
    return results;
  }
};

// Stack: regex → your domain detector → LLM for everything else
const proxy = new PrivacyProxy({
  detectors: [
    ...defaultDetectors,
    germanInsuranceDetector,
    new LlmDetector({ model: 'qwen3:1.7b' }),
  ],
});
```

You can also add custom generators for your entity types:

```typescript
const proxy = new PrivacyProxy({
  detectors: [...defaultDetectors, new LlmDetector()],
  generators: {
    // Custom replacement for your entity type
    insurance_id: (real) => 'X' + Math.random().toString().slice(2, 11),
  },
});
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

## Examples

### Health data with local LLM ([examples/health-data.ts](examples/health-data.ts))

Full round-trip — local LLM detects patient names and providers, Claude analyzes the masked record, unmask restores real data:

```bash
export ANTHROPIC_API_KEY=sk-...
bun run examples/health-data.ts
```

### Anthropic SDK integration ([examples/anthropic-agent.ts](examples/anthropic-agent.ts))

```bash
export ANTHROPIC_API_KEY=sk-...
bun run examples/anthropic-agent.ts
```

## Model benchmarks

Detection rates on a realistic clinical patient record containing 10 PII entities (names, DOB, medical record number, insurance ID, organization, address, email, phone). Regex detectors handle email + phone in all configurations — the LLM layer adds semantic entity detection.

| Model | Size | Entities detected | Notes |
|---|---|---|---|
| Regex only | 0 | 2/10 | Email + phone only — no names, orgs, or IDs |
| + `qwen3:0.6b` | 522 MB | 6/10 | Catches patient name, DOB, insurance, address. Misses doctor names, org, medical record |
| + `llama3.2` | 2.0 GB | 3/10 | Poor structured output — mostly finds just the patient name |
| **+ `qwen3:1.7b`** | **1.4 GB** | **9/10** | **Catches all patient PII, org, address, medical record. Misses one buried doctor name** |

**Recommendation:** `qwen3:1.7b` is the default — best accuracy-to-size ratio. All models produce **perfect round-trip** (unmask restores the original text exactly).

Test input:
```
Patient: Marcus Weber (DOB: 15.03.1987)
MRN: MRN-2024-08391
Provider: Dr. Sarah Chen, Universitätsklinikum Heidelberg
Insurance: TK 109876543
Contact: marcus.weber@gmail.com, +49 170 1234567
Address: Hauptstraße 42, 68161 Mannheim, Germany
Referred by Dr. Anika Hoffmann, Hausarztpraxis Mannheim.
```

## Roadmap

- [x] **v0.1** — Regex detection, faker replacement, bijective round-trip
- [x] **v0.2** — Pluggable entity detection — bring your own detectors (local LLM, custom regex). Layered pipeline: fast regex first, LLM for names/locations/domain-specific entities
- [ ] **v0.3** — Tool-aware selective masking (keep location real for hotel search, mask for email)
- [ ] **v0.4** — Persistent map backends (Redis, SQLite)
- [ ] **v0.5** — Anthropic/OpenAI SDK middleware (drop-in agent integration)

## License

MIT
