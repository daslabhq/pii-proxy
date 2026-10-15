# pii-proxy

[![Test](https://github.com/daslabhq/pii-proxy/actions/workflows/test.yml/badge.svg)](https://github.com/daslabhq/pii-proxy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Privacy layer for AI agents. Mask PII before it reaches any LLM — unmask when writing back to your systems. PII detection runs locally and never leaves your infrastructure.

```
Your data ──→ [pii-proxy] ──→ LLM sees only fake data ──→ [pii-proxy] ──→ Real data restored
              mask()           plausible fakes                unmask()       perfect round-trip
```

Works with Node.js, Bun, and any OpenAI-compatible API (Claude, GPT, local models).

**Local detection.** Fine-tuned `gliner_small-v2.1` on the full 55-label Nemotron-PII: **91.5% fine F1 — matches NVIDIA's flagship `gliner-PII` (90.9%) on the identical held-out records**, from a base model a third the size (582 MB vs 1.7 GB), with a reproducible recipe. On out-of-distribution data it beats default Microsoft Presidio **75.7% to 42.0% F1** ([exp 006](experiments/006-presidio-head-to-head/)). Runs locally, no cloud dependency. [Reproducible benchmarks →](experiments/)

## Why

Your AI agent processes patient records, insurance claims, customer data. You don't want real names, emails, and ID numbers hitting Claude or GPT. But token-based masking (`PERSON_1`, `EMAIL_2`) degrades fluency — LLMs lose track of meaningless placeholders across long contexts.

**pii-proxy** replaces PII with plausible fake values — the model parses realistic-looking text fluently, and a bijective map reverses every fake when you write back. (This keeps the model fluent but means it reasons over the fakes — see [When this works](#when-this-works-and-when-it-doesnt) for failure modes.)

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
// → "Ship order to jordan.smith@example.net, tracking BFUI0000482918EZ"

// Send masked.text to your LLM...

// Reverse all fakes back to real values
const real = proxy.unmask(llmResponse);
// "I'll notify jordan.smith@example.net" → "I'll notify alex@example.com"
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

Like-for-like replacement keeps the model's input fluent. The model parses realistic-looking text without losing entity tracking over many opaque tokens — but it's reasoning about the *fake*, not the real.

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
| Email | Regex | Fake at a reserved example domain — it can never reach anyone |
| Phone | Regex | Format-preserving fake |
| Credit card | Regex + Luhn | Valid fake card number |
| IP address | Regex (IPv4 + IPv6) | Random valid IP, same family |
| UUID | Regex | Random UUID |
| URL | Regex | Sanitized URL |
| Tracking number | Regex (UPS, USPS, DHL, etc.) | Format-preserving fake |
| Passport number | Context-gated regex ("passport" + ID token) | Format-preserving fake |
| National ID / SSN | Context-gated regex | Format-preserving fake |
| Driver license | Context-gated regex | Format-preserving fake |
| ID card | Context-gated regex | Format-preserving fake |
| Person name | Local LLM | Faker name |
| Organization | Local LLM | Faker company |
| Location | Local LLM | Faker address/city |
| Date of birth | Local LLM | Format-preserving fake date |
| Medical record | Local LLM | Format-preserving fake |
| Insurance ID | Local LLM | Format-preserving fake |
| First / last name | Table columns (`pii-proxy/files`) | Faker name; follows a title's sex |
| Employee ID | Table columns (`pii-proxy/files`) | Letters and first two digits kept, rest new |
| *Custom* | Local LLM | Format-preserving fallback |

Whatever generates it, a fake never equals or contains a real value the proxy has seen — so searching masked output for real names is a valid check.

## Structured data

Mask entire objects (e.g., tool call inputs):

```typescript
const { masked } = await proxy.maskObject({
  to: "alex@example.com",
  subject: "Order update",
  body: "Tracking: AETH0000345323DY",
  metadata: { ip: "10.0.0.1" }
});

// masked.to → "jordan.smith@example.net"
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

## Packs

A pack bundles what one locale or domain needs — detectors, generators, column headers, titles — and plugs in with `use`. A pack's detectors run before the defaults, so its typed IDs win over looser matches on the same span. Without packs, pii-proxy is English.

```typescript
import { PrivacyProxy } from 'pii-proxy';
import { thai } from 'pii-proxy/packs/th';

const proxy = new PrivacyProxy({ use: [thai()] });
```

| Pack | Kind | Adds |
|---|---|---|
| [`pii-proxy/packs/th`](src/packs/th/README.md) | locale | Thai names, national IDs, phone numbers, column headers and titles |
| [`pii-proxy/packs/hr`](src/packs/hr/README.md) | domain | Employee codes, sign-off columns; addresses and bank accounts withheld |

Locale packs own **words** — names, national ID formats, titles, what a column is called in that language. Domain packs own **rules** — which columns identify someone in that field and how each is treated. So they combine without a pack per pair: `use: [thai(), hr()]`.

Writing one is a function returning `{ detectors?, generators?, headers?, titles? }` — see [`src/packs/th`](src/packs/th/index.ts). Generators receive a context: `attempt` (widen the range when an earlier fake was taken), a seeded `rng`, `hints` (e.g. the sex a title implies) and `count` (values of this type so far, for numbered labels such as `withheld('address')` → `[address withheld #3]`).

## Masking files (`pii-proxy/files`)

Spreadsheets and CSV are masked column by column; text goes through the detectors plus every value the map already holds.

```typescript
import { PrivacyProxy } from 'pii-proxy';
import { hr } from 'pii-proxy/packs/hr';
import { maskFiles } from 'pii-proxy/files';

const proxy = new PrivacyProxy({ use: [hr()] });
const { results, skipped } = await maskFiles([
  { name: 'payroll-09.xlsx', bytes: payroll },
  { name: 'bank-09.csv', bytes: bank },
], proxy);
// results[0].files   → [{ name: 'payroll-09 (masked).csv', mimeType: 'text/csv', bytes }]
// results[0].columns → [{ header: 'EmployeeId', type: 'employee_id' }, { header: 'FirstName', type: 'first_name' }, …]
```

- **Columns by header, then by values.** `FirstName`, `Surname`, `Email`, `Phone` — plus any headers a pack adds — name a column's type; a column whose header says nothing is typed when most of its values are emails, phones or national IDs.
- **Everything else untouched.** Departments, positions, amounts and dates stay exactly as they were, so totals computed from the masked copy equal the originals.
- **Full names rebuilt from their parts.** `Mr. John Smith` becomes the same row's title with its masked first and last name, in the same layout, and the pairing is linked so the whole name is caught in free text too.
- **Batches are consistent.** Every personal value of a batch is reserved before any fake is made; with the map persisted, next month's files get the same fakes.
- **One CSV per sheet.** A workbook comes back as CSV, with a BOM so Excel opens non-Latin text correctly.

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
- **The bijective map is sensitive.** It maps real values to fakes — treat it like the data itself. Encrypt at rest, scope per session, and control access. Use `proxy.getMap().serialize()` for persistence; the format is a JSON array of `[real, fake]` pairs, with the type third when it is known.
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

### Open PII models in six languages, on long letters and on Thai ([exp 007](experiments/007-detector-benchmark/))

Before shipping a model detector by default: `pii-proxy-ner` against
[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter),
[GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi),
NVIDIA's `gliner-PII` and three more. Share of personal data fully masked
(every letter and digit inside a detection, of any type):

| | identifiers, 6 languages¹ | everything ai4privacy labels, EN / other 5 | SPY (long letters) | Thai² | CPU, 2 cores |
|---|---|---|---|---|---|
| OpenAI Privacy Filter | **95–98%** | 80.4% / 78–81% | **86.7%** | 70.9% | 2.6 s |
| knowledgator gliner-pii-base | 84–88% | **89.1% / 86–89%** | 83.0% | — | 1.5 s |
| GLiNER2-PII | 76–85% | 86.9% / 80–86% | 82.0% | **91.5%** | 2.6 s |
| NVIDIA gliner-PII | 73–85% | 87.9% / 79–85% | 82.8% | 42.1% | 7.1 s |
| **`pii-proxy-ner`** | 68–83% | 85.4% / 73–80% | 83.8% | 33.9% | **1.3 s** |

¹ names, emails, phones, ID numbers, IPs, usernames, passwords.
² with a Thai word segmenter in front of the model.

The filter masks nearly every identifier in every language and leaves times
of day, states, countries and gender. knowledgator masks the most overall, but
a quarter of its flags are not PII. `pii-proxy-ner` is the fastest and ties
NVIDIA on Nemotron, but trails it on ai4privacy and drops outside English.
On Thai GLiNER2-PII leads by 20 points, once the text is split into words. GLiNER v1
checkpoints (ours, NVIDIA's) stop reading after 384 tokens, about 300 words:
window long text. Full tables, typed F1, the scoring conventions and why they
matter, speed, size and licences: [exp 007](experiments/007-detector-benchmark/).

### vs Microsoft Presidio, out of distribution ([exp 006](experiments/006-presidio-head-to-head/))

The comparison that matters for choosing a tool: both systems on a dataset
**neither was tuned on** — ai4privacy `pii-masking-300k`, English, 1,000
records / 6,357 annotated spans. Entity-level F1, plus *coverage* (% of PII
spans masked by any detection — the leak-prevention number):

| System | F1 | Coverage |
|---|---|---|
| **pii-proxy** (`pii-proxy-ner` + regex backstop) | **75.7%** | **89.0%** |
| pii-proxy (model only) | 74.8% | 83.6% |
| Presidio (default, spaCy `en_core_web_lg`) | 42.0% | 53.8% |
| pii-proxy (regex layer only) | 32.0% | 27.6% |

`pii-proxy-ner` is the exp-005 model below. Restricted to the 9 entity
categories both systems support (removing our vocabulary advantage):
**80.5% vs 49.1%** — pii-proxy wins 13 of 15 categories; Presidio narrowly
keeps IP and passport, and ties on email. Regex-vs-regex (Presidio's
NAME/LOCATION/DATETIME come from spaCy NER, not patterns): ours 32.0% vs
their pattern recognizers 27.8%. Presidio is benchmarked untuned, as in
virtually every published comparison — treat its numbers as a floor. Full
methodology, fairness rules, per-type tables, ablations, and an adversarial
audit of the benchmark itself: [exp 006](experiments/006-presidio-head-to-head/).

> Note: the npm package ships the regex layer + LLM detector today; the
> fine-tuned model runs via the experiments pipeline. Packaging it as a
> drop-in detector is the current roadmap priority — exp 006 measures
> exactly what that's worth (32.0% → 75.7%).

### Same-records head-to-head ([exp 005](experiments/005-nvidia-baseline/))

Fine-tuning `gliner_small-v2.1` on the **full 55-label Nemotron-PII** vocabulary **matches NVIDIA's flagship `gliner-PII` from a base model a third the size** — both scored on the *identical* held-out records, fine F1:

| Model | Base | Fine F1 | Precision | Recall |
|---|---|---|---|---|
| **`pii-proxy` (`gliner_small-v2.1`)** | 153M parameters | **91.5%** | 90.2% | 92.8% |
| `nvidia/gliner-PII` | 445M parameters | 90.9% | 90.6% | 91.3% |

Both models on the **same** 500-record held-out `test` set (hash-verified identical), each at its best native-label query, threshold 0.5, leakage-checked 0/500 — the eval is [`head_to_head_eval.py`](experiments/head_to_head_eval.py), both query strings published. **Read +0.6pp as a statistical tie** (exp 007's rerun on the same records puts the difference's 95% interval at −0.1 to +1.3pp), not "we beat NVIDIA" — the real claim is *flagship quality from a base a third the size, with a reproducible recipe* (NVIDIA shipped weights, no recipe). On the healthcare slice the order reverses by a hair: NVIDIA 91.0%, pii-proxy 90.8%. **Caveat:** both models trained on Nemotron, so this measures in-distribution reproduction — the out-of-distribution result ([exp 006](experiments/006-presidio-head-to-head/)) is where the fine-tune earns its keep.

### Healthcare subset head-to-head ([exp 004](experiments/004-finetune-fine-labels/))

> **The canonical healthcare number is in the head-to-head above** — clean held-out
> `test` split, NVIDIA 91.0% / pii-proxy 90.8%. The 96%+ figures below come from an
> earlier, smaller healthcare *subset* eval that does **not** reproduce on the clean
> split; read them as a ceiling on an easier cut, not the headline. Kept here for the
> query-strategy and latency detail, which still hold.

Evaluation on the Nemotron-PII healthcare subset, same held-out 100-record test set across all methods. All numbers independently verified ([`verify.py`](experiments/003-finetune-gliner-small/verify.py)).

**Truly fair head-to-head** — same labels, same vocabulary, same query strategy:

| Method | Fine F1 | Coarse F1 | Latency | Weights |
|---|---|---|---|---|
| **nvidia/gliner-PII** | **96.2%** | **96.7%** | 211ms | 1699MB |
| Our fine-tuned `gliner_small` ([exp 004](experiments/004-finetune-fine-labels/)) | 94.9% | 96.1% | 144ms | 582MB |
| Our BERT classifier ([exp 002](experiments/clean_benchmark.py)) | — | 93.9% | 26ms | 438MB |
| Claude Sonnet (proper prompt) | — | 92.5% | 2.5s | cloud |
| `gliner_small-v2.1` (zero-shot baseline) | 54.8% | — | 115ms | 582MB |

**The honest takeaway:** NVIDIA's flagship is still slightly more accurate. We're 1.3pp behind at fine granularity and 0.6pp behind at coarse granularity — but 1.5x faster and on a 3x smaller base model. The BERT classifier is the fastest option at 26ms when zero-shot capability isn't needed.

**Three valid optima depending on your constraints:**

```
Need maximum accuracy?         nvidia/gliner-PII   (96.7% coarse, 211ms, 1.7GB)
Need balance + zero-shot?      Our gliner_small     (96.1% coarse, 144ms, 582MB)
Need sub-30ms latency?         Our BERT classifier  (93.9% coarse,  26ms, 438MB)
```

**The compile pipeline works** — `gliner_small-v2.1` (54.8% baseline) → 96.1% coarse F1 after 15 minutes of fine-tuning on a laptop. Within striking distance of NVIDIA's flagship.

**Methodology pitfall (worth knowing):** GLiNER bi-encoder models are extremely sensitive to query string choice. NVIDIA's `gliner-PII` scores 79.8% with simple coarse queries ("person_name"), 90.4% with natural-language ("person name"), and 97.2% with fine native labels ("first_name", "last_name"). The same model — three very different scores. Anyone benchmarking GLiNER must report the query strategy. See [`verify_labels.py`](experiments/003-finetune-gliner-small/verify_labels.py).

**Recommended training pattern:** train on fine labels, collapse to coarse post-hoc. Beats training directly on coarse (96.1% vs 95.5% in our experiments — richer supervision signal during training).

See [`experiments/`](experiments/) for every script, every result, every caveat.

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
- [x] **Model: training recipe** — reproducible fine-tune pipeline ([exps 002–005](experiments/)): `pii-proxy-ner` matches NVIDIA's flagship `gliner-PII` on Nemotron with a third of its parameters (153M vs 445M); a 3-epoch run of the recipe takes 39 min for ~$8. `pii-proxy-learn` CLI labels your own traces for domain fine-tunes
- [x] **Model: validation** — out-of-distribution pipeline benchmark vs Microsoft Presidio ([exp 006](experiments/006-presidio-head-to-head/)): 75.7% vs 42.0% F1, with adversarial audit of the methodology
- [x] **Packs and files** — locale packs (`pii-proxy/packs/*`), spreadsheet and CSV masking (`pii-proxy/files`), typed map entries, fakes that never contain a real value
- [ ] **v0.3** — Ship a model detector by default (ONNX export, model-first default ordering). Benchmarked in [exp 007](experiments/007-detector-benchmark/): OpenAI Privacy Filter masks the most identifiers in six languages (95–98%), GLiNER2-PII leads on Thai with word segmentation (91.5%), `pii-proxy-ner` is the fastest (1.3 s per 450-word letter on 2 CPU cores) and needs its 55-label checkpoint published. Choosing between them is next
- [ ] **Pack adapters** — a pack may carry a LoRA adapter for the model detector (GLiNER2 loads and swaps 2–10 MB adapters natively). Locale packs own the adapter (language is what a base model lacks); domain packs stay labels plus rules (zero-shot handles domain concepts), so `use: [thai(), medical()]` is one pass, not two. First: Thai. [exp 007](experiments/007-detector-benchmark/) measured it: with a word segmenter in front, GLiNER2-PII masks 91% of Thai names without Thai fine-tuning (50% without the segmenter), so segmentation comes before any adapter
- [ ] **Domain packs, when there is data to build them against** — medical (record and insurance numbers, dates shifted consistently per patient, ages 90+ grouped), legal (case and docket numbers, parties named in captions)
- [x] **Regex upgrades from exp 006's gap list** — IPv6 (IP F1 61%→96%) and context-gated ID detection (passport/SSN/driver-license/ID-card via keyword + token; regex layer 20.6%→32.0%, beating Presidio's pattern recognizers at 27.8%)
- [ ] **v0.4** — Detection quality from exp 006 findings: type/confidence-aware span conflict resolution (replaces first-wins merge), locale-aware ID regexes
- [ ] **v0.5** — Tool-aware selective masking (keep location real for hotel search, mask for email)
- [ ] **v0.6** — Persistent map backends (Redis, SQLite)
- [ ] **v0.7** — Anthropic/OpenAI SDK middleware (drop-in agent integration)

## License

MIT — built by [Daslab](https://github.com/daslabhq).
