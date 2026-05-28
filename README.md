# pii-proxy

[![Test](https://github.com/daslabhq/pii-proxy/actions/workflows/test.yml/badge.svg)](https://github.com/daslabhq/pii-proxy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Privacy layer for AI agents. Mask PII before it reaches any LLM — unmask when writing back to your systems. PII detection runs locally and never leaves your infrastructure.

```
Your data ──→ [pii-proxy] ──→ LLM sees only fake data ──→ [pii-proxy] ──→ Real data restored
              mask()           plausible fakes                unmask()       perfect round-trip
```

Works with Node.js, Bun, and any OpenAI-compatible API (Claude, GPT, local models).

**Local detection.** Fine-tuned BERT classifier: **93.9% F1, 26ms/record** on the NVIDIA Nemotron-PII healthcare benchmark. Fine-tuned GLiNER-small: **95.5% F1, 126ms/record** (zero-shot capable). Both run on a CPU with no cloud dependency. [Reproducible benchmarks →](experiments/)

## Why

Your AI agent processes patient records, insurance claims, customer data. You don't want real names, emails, and ID numbers hitting Claude or GPT. But token-based masking (`PERSON_1`, `EMAIL_2`) degrades fluency — LLMs lose track of meaningless placeholders across long contexts.

**pii-proxy** replaces PII with plausible fake values — the model parses realistic-looking text fluently, and a bijective map reverses every fake when you write back. (Fluency, not correctness — see [When this works](#when-this-works-and-when-it-doesnt) for failure modes.)

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

**v0.2** adds a local LLM detection layer. A model running on your machine (via [Ollama](https://ollama.com)) detects names, organizations, locations, and domain-specific entities. **PII never leaves your network** — not even for detection.

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

**Detector order = priority.** Each detector returns `Detection[]` (or `Promise<Detection[]>` for async). The first detector to claim a span wins; later detectors that overlap that span are dropped. Put your most-specific detectors first.

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

## When this works (and when it doesn't)

Like-for-like replacement preserves **fluency, not correctness**. The model parses realistic-looking text without losing entity tracking over many opaque tokens — but it's reasoning about the *fake*, not the real.

**Works well for:**
- Drafting, replying, summarization, extraction, routing
- Multi-entity tracking where opaque `PERSON_1` tokens degrade attention
- Any task where the entity is a *referent*, not analyzed for its surface properties

**Breaks (silently) for:**
- **Surface inference.** The model infers from the fake's surface — locale, gender, demographics. `Marcus Weber` → `Mei Chen` is a legal swap; "draft this in the patient's likely language" picks the wrong one.
- **Cross-entity coherence.** `Marcus Weber` and `Anna Weber` get severed; the model loses the family relationship.
- **Generated new PII.** The model can invent associated names ("Dr. Schmidt") that were never in the map — unmask leaves them in, and hallucinated PII leaks through.

If your task leans on entity surface properties, treat pii-proxy as a fluency layer, not a correctness layer. For high-stakes inference, defense in depth: pii-proxy + structured output schema + post-hoc validation.

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

## Security model

pii-proxy is designed so that **real PII never reaches the cloud LLM**.

**Data flow:**

```
┌─────────────────────────────────────────────────────┐
│  Your infrastructure (on-prem / VPC)                │
│                                                     │
│  Real data ──→ Regex detection (in-process)         │
│            ──→ Local LLM detection (Ollama, local)  │
│            ──→ Fake replacement (in-process)         │
│                        │                            │
│                        ▼                            │
│              Masked data (fakes only)               │
└────────────────────────┬────────────────────────────┘
                         │ only fake data crosses this boundary
                         ▼
               ┌──────────────────┐
               │  Cloud LLM API   │
               │  (Claude, GPT)   │
               └──────────────────┘
```

- **Detection is local.** Regex runs in-process. The LLM detector calls a model on your machine or your private network — never a cloud API.
- **The bijective map is sensitive.** It maps real values to fakes — treat it like the data itself. Encrypt at rest, scope per session, and control access. Use `proxy.getMap().serialize()` for persistence; the format is a JSON array of `[real, fake]` pairs.
- **Unmask is deterministic.** Same map always produces the same reversal. No network calls, no side effects.
- **Round-trip integrity.** Every `mask()` → `unmask()` cycle restores the original text exactly. This is tested on every commit.

**What pii-proxy does NOT do:**
- It does not guarantee 100% PII detection — regex has known patterns, the LLM layer catches most names/orgs/locations, but novel entity types may slip through. Defense in depth is recommended.
- It does not encrypt the map for you — integrate with your existing secrets management (Vault, KMS, encrypted storage).
- It does not log or audit automatically — call `proxy.getMap().entries()` to inspect or log what was masked per session.

## Persistence

> **⚠ The map IS the PII.** It maps every real value to its fake — anyone with the map can reverse every masked record. Encrypt before storing. See [Security model](#security-model).

Save and restore the map across sessions:

```typescript
// Save — encrypt the serialized map before storing
const data = proxy.getMap().serialize();
await redis.set('pii-session:123', encrypt(data));  // bring your own encryption (Vault, KMS, libsodium)

// Restore in a new process
const proxy2 = new PrivacyProxy();
proxy2.loadMap(decrypt(await redis.get('pii-session:123')));
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

## Benchmarks

Full evaluation on NVIDIA's [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) healthcare subset (725 records, 4,703 labeled spans, same held-out 100-record test set across all methods). Scripts in [`experiments/`](experiments/).

**Honest summary:** NVIDIA's `gliner-PII` (their flagship, 1.7GB) is the strongest detector at **97.2% F1**. Our fine-tuned `gliner_small-v2.1` (582MB, 3x smaller) reaches **95.5% F1** and is 1.7x faster. Our fine-tuned BERT classifier is the fastest at **26ms/record, 93.9% F1**.

| Method | F1 | Latency/record | Weights | Zero-shot capable |
|---|---|---|---|---|
| **nvidia/gliner-PII** (zero-shot, native labels) | **97.2%** | 210ms | 1699MB | Yes |
| Our fine-tuned `gliner_small-v2.1` ([exp 003](experiments/003-finetune-gliner-small/)) | 95.5% | 126ms | 582MB | Yes |
| Our fine-tuned BERT classifier ([exp 002](experiments/clean_benchmark.py)) | 93.9% | 26ms | 438MB | No |
| Claude Sonnet (proper prompt) | 92.5% | 2.5s | cloud API | — |
| `gliner_small-v2.1` (zero-shot baseline) | 54.8% | 115ms | 582MB | Yes |

All numbers from independent verification on the same test set (fingerprint `ff2fa10db2eb0b55`, zero train/test leakage).

**Three valid optima depending on your need:**

```
Need maximum accuracy?         nvidia/gliner-PII (97.2%, 210ms, 1.7GB)
Need fast + accurate + small?  Our fine-tuned gliner_small (95.5%, 126ms, 582MB)
Need sub-30ms latency?         Our fine-tuned BERT (93.9%, 26ms, 438MB)
```

**The compile pipeline works.** Fine-tuning `gliner_small-v2.1` on just 575 records of healthcare PII (15 min on Apple MPS) takes the base model from 54.8% → 95.5% F1. Within 1.7pp of NVIDIA's flagship using 1/3 the parameters.

**Methodology note:** GLiNER bi-encoder models are sensitive to query string choice. NVIDIA's `gliner-PII` scores 90.4% with natural-language labels ("person name") and 97.2% with native training labels ("first_name"). Always benchmark with the model's native vocabulary. See [`verify_labels.py`](experiments/003-finetune-gliner-small/verify_labels.py).

See [`experiments/`](experiments/) for full reproducibility — every script, every result, every caveat.

## Comparison with alternatives

| | pii-proxy | Presidio | Private AI | Nightfall |
|---|---|---|---|---|
| Data leaves your infra | **No** | No | Yes | Yes |
| Round-trip unmask | **Yes** | No | No | No |
| Replacement | Plausible fakes | Tokens (`<PERSON>`) | Tokens | Tokens |
| Custom entity types | Pluggable detectors | Custom recognizers | Limited | Limited |
| License | MIT | MIT | Commercial | Commercial |

## Roadmap

- [x] **v0.1** — Regex detection, faker replacement, bijective round-trip
- [x] **v0.2** — Pluggable entity detection — bring your own detectors (local LLM, custom regex). Layered pipeline: fast regex first, LLM for names/locations/domain-specific entities
- [ ] **v0.3** — Tool-aware selective masking (keep location real for hotel search, mask for email)
- [ ] **v0.4** — Persistent map backends (Redis, SQLite)
- [ ] **v0.5** — Anthropic/OpenAI SDK middleware (drop-in agent integration)

## License

MIT — built by [Daslab](https://github.com/daslabhq).
