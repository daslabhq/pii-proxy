# Experiment 007: open PII detectors in six languages, on long letters and on Thai

v0.3 puts a model detector in front of the regex layer by default. Before
picking one, this measures the open contenders on one harness:
[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter),
[GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
(Fastino), NVIDIA's [`gliner-PII`](https://huggingface.co/nvidia/gliner-PII),
our `pii-proxy-ner`, two more GLiNER PII models, Presidio and our regex layer.
The data: ai4privacy's validation split in six languages, SPY (long synthetic
legal and medical letters), the Nemotron-PII held-out set, and a Thai set
built for this.

## Result

**What a proxy needs to know is what gets through.** The headline number is
the share of personal data *fully masked*: every letter and digit of a gold
span inside something the model flagged, whatever it called it. Typed F1 is
reported too, and it ranks the same models differently for reasons that have
little to do with leaks ([How it is scored](#how-it-is-scored)).

| Question | Answer |
|---|---|
| Which model masks the most identifiers? | **OpenAI Privacy Filter**: 95.4–97.8% of names, emails, phones, ID numbers, IPs, usernames and passwords in each of six languages. The GLiNER models: 67.6–87.8%. |
| What does the filter leave? | Times of day (6% masked), states (4%), countries (0%) and gender (11%). It masks dates, birth dates, cities and street addresses like the others. ai4privacy counts all four as PII, so its overall share is lower: 78–81%. |
| Which masks the most overall? | knowledgator `gliner-pii-base`: 86–89% in every language, but a quarter of what it flags is not PII (69–77% of its flags land on PII). GLiNER2-PII and NVIDIA: 79–88%, with 85–88% of flags on PII. |
| Outside English? | GLiNER2-PII, NVIDIA and knowledgator stay within a few points of English. `pii-proxy-ner` drops 5–13 points: its base model is English. |
| Long letters (SPY, English)? | The filter 86.7%; GLiNER2-PII, NVIDIA, knowledgator and `pii-proxy-ner` 82.0–83.8%. The gap is personal URLs (filter 71%, the rest about 24%). |
| Long documents in general? | GLiNER v1 checkpoints with `max_len` 384 (NVIDIA's, ours, gliner_multi) stop reading after about 300 words. Handed whole SPY letters, NVIDIA's model masks 54.4%; in 200-word windows, 82.8%. F1 does not move. |
| Thai? | **GLiNER2-PII with a word segmenter in front: 91.5%** (names 91.1%), never fine-tuned on Thai. Without the segmenter, 69.4% (names 50.0%). Next best: the filter, 70.9%. NVIDIA 42.6%, `pii-proxy-ner` 33.9%. |
| Does `pii-proxy-ner` still match NVIDIA's? | On Nemotron, yes: 92.5% vs 91.8% fine F1, a tie (difference's 95% interval −0.1 to +1.3pp). On ai4privacy English NVIDIA is ahead (masked +2.5pp, interval +1.3 to +3.8), and in the other five languages by 4.6–6.1pp. On SPY ours masks 1.0pp more. |
| Speed, one ~450-word letter, 2 CPU cores, full label set | `pii-proxy-ner` **1.3 s**, knowledgator 1.5 s, gliner_multi 2.4 s, the filter 2.6 s, GLiNER2-PII 2.6 s, NVIDIA 7.1 s. Asking for 5 labels instead of all makes each GLiNER model about a quarter faster. |
| Runs from TypeScript today? | The filter: official ONNX, Transformers.js. knowledgator: official ONNX, GLiNER.js. GLiNER v1 checkpoints (ours, NVIDIA's, gliner_multi): GLiNER.js after an ONNX export. **GLiNER2-PII: no official ONNX export or JS runtime.** |

## How it is scored

Per system and dataset, from [`score.py`](score.py):

- **masked**: share of gold spans whose every letter and digit lies inside
  some detection, of any type, before any flattening (a masker masks every
  span it is given). The headline. It does not depend on label names or on
  where a dataset draws a span's boundaries.
- **flags that are PII** (`flagged_p`): share of detections, of any type,
  that overlap some gold span. Masked's precision: what a proxy would mask
  that was not personal data. Meaningless on SPY (below), meaningful elsewhere.
- **exact F1** (bucket and both boundaries) and **overlap F1** (bucket and any
  overlap, exp 006's primary metric), each with greedy one-to-one matching.

Three conventions in the typed scores move them by many points, so they are
stated here and applied to every model alike:

1. **Name parts are merged where gold labels whole names** (SPY, Thai). The
   Nemotron-trained models tag "Paul" and "Acosta" separately; exact match
   counts that as one miss and two false positives. Merging adjacent name
   parts moves NVIDIA's SPY exact F1 from 36.5% to 50.6% and ours from 36.6%
   to 51.0%. The unmerged numbers are kept under `names_as_returned`.
   GLiNER2-PII's nested spans (person ⊃ full_name ⊃ first_name) are flattened
   for the same reason (34.7% → 50.1%).
2. **The OpenAI filter's eight categories mean what its model card says**:
   `account_number` covers every ID number, `secret` covers passwords,
   `private_url` covers IP addresses, `private_person` covers usernames.
   Mapped narrowly, its ai4privacy overlap F1 was 53.7%; mapped as the card
   maps them, 80.1%.
3. **Every model is asked for what the dataset labels**, in its own label
   names, plus a plain zero-shot label for anything it has no name for. On
   ai4privacy and Nemotron that includes `time` and `date` even when a model
   already has a date-of-birth label: times are the largest label in
   ai4privacy (496 spans), and a model never asked for them cannot find them.

The masked number needs none of the three.

A first pass of this experiment ranked by exact F1 without conventions 1–3,
and reported GLiNER2-PII 10pp ahead of NVIDIA and `pii-proxy-ner` on SPY and
the filter at 53.7% on ai4privacy. Both gaps came from the conventions, not
from what the models found; an adversarial review of the harness caught them.

## Six languages (ai4privacy)

`pii-masking-300k` validation split: exp 006's 1,000 English records, and
1,000 records each of German, French, Italian, Spanish and Dutch drawn the
same way (seed 42). Exp 006's buckets.

### Share fully masked, model alone

| Model | SPY | EN | DE | FR | IT | ES | NL | Nemotron | Thai¹ |
|---|---|---|---|---|---|---|---|---|---|
| GLiNER2-PII | 82.0% | 86.9% | 85.4% | 85.9% | 84.1% | 84.9% | 80.1% | 88.6% | **91.5%** |
| OpenAI Privacy Filter | **86.7%** | 80.4% | 79.7% | 81.0% | 79.1% | 78.3% | 78.3% | 73.4% | 70.9% |
| NVIDIA gliner-PII | 82.8% | 87.9% | 84.6% | 85.3% | 81.4% | 83.8% | 78.6% | 94.8% | 42.1% |
| `pii-proxy-ner` | 83.8% | 85.4% | 80.0% | 79.9% | 75.5% | 77.7% | 72.9% | **95.8%** | 33.9% |
| knowledgator gliner-pii-base | 83.0% | **89.1%** | **89.3%** | **87.9%** | **85.9%** | **87.5%** | **86.0%** | 84.8% | — |
| gliner_multi_pii-v1 | 77.0% | 71.5% | 72.8% | 69.6% | 66.3% | 68.1% | 68.0% | 78.0% | 53.5% |
| `daslabhq/pii-proxy` (24-label) | 68.2% | 64.1% | 51.9% | 53.1% | 52.4% | 51.7% | 50.4% | 79.6% | — |
| Presidio (default) | 64.4% | 48.0% | — | — | — | — | — | 79.4% | — |
| regex layer only | 24.0% | 25.6% | 14.5% | 18.5% | 17.3% | 17.7% | 18.9% | 22.0% | 34.9% |

¹ segmented (see Thai). Nemotron is in-distribution for NVIDIA and `pii-proxy-ner`.

### Identifiers and context

Identifiers: names, emails, phones, ID numbers, IP addresses, usernames,
passwords, coordinates. Context: dates, times, places, titles, gender. The
split is ours; the per-type tables in `results.json` let you draw it elsewhere.

| Model | EN | DE | FR | IT | ES | NL | flags that are PII |
|---|---|---|---|---|---|---|---|
| OpenAI Privacy Filter | **97.5%** / 60.3% | **97.8%** / 59.3% | **97.5%** / 62.8% | **95.4%** / 60.8% | **97.4%** / 57.4% | **95.5%** / 57.6% | **93–94%** |
| knowledgator gliner-pii-base | 87.0% / 91.6% | 86.0% / 93.0% | 86.8% / 89.0% | 85.3% / 86.6% | 87.8% / 87.1% | 83.7% / 88.8% | 69–77% |
| GLiNER2-PII | 85.4% / 88.7% | 84.2% / 86.8% | 84.0% / 88.0% | 82.3% / 86.1% | 84.1% / 85.7% | 75.8% / 85.2% | 86–88% |
| NVIDIA gliner-PII | 85.4% / 90.8% | 81.6% / 88.1% | 81.7% / 89.3% | 80.3% / 82.7% | 82.9% / 84.9% | 73.1% / 85.3% | 85–88% |
| `pii-proxy-ner` | 82.7% / 88.5% | 77.0% / 83.5% | 76.3% / 83.9% | 73.9% / 77.3% | 75.0% / 80.6% | 67.6% / 79.2% | 78–88% |
| gliner_multi_pii-v1 | 73.7% / 68.8% | 77.3% / 67.6% | 73.8% / 65.0% | 71.0% / 61.1% | 74.9% / 60.7% | 71.6% / 63.7% | 74–79% |

identifiers masked / context masked.

**The GLiNER models do not flag IPv6 addresses.** In English, GLiNER2-PII
masks all 127 IPv4 addresses and none of the 147 IPv6 ones, NVIDIA 123 and
none; neither flags any part of one. The filter masks all 147. The regex layer
catches them (94–99% of IPs per language), and NVIDIA + regex reaches 95–99%.

**knowledgator's extra flags** come mostly from the `title` label, which it
puts on job titles and plain nouns ("Forscher", "Benutzer").

### Typed scores, English

| System | masked | 95% | overlap P | overlap R | overlap F1 | 95% | exact F1 |
|---|---|---|---|---|---|---|---|
| knowledgator gliner-pii-base | **89.1%** | 87.8–90.4% | 67.0% | 77.3% | 71.8% | 70.1–73.4% | 67.3% |
| NVIDIA gliner-PII | **87.9%** | 86.6–89.2% | 75.5% | 77.6% | 76.6% | 75.0–78.2% | 73.7% |
| GLiNER2-PII | **86.9%** | 85.7–88.1% | 73.5% | 73.5% | 73.5% | 71.8–75.1% | 68.8% |
| `pii-proxy-ner` | **85.4%** | 83.9–86.9% | 76.3% | 74.2% | 75.3% | 73.7–76.7% | 71.9% |
| OpenAI Privacy Filter | **80.4%** | 79.1–81.7% | 87.9% | 73.5% | **80.1%** | 78.9–81.2% | 76.1% |
| gliner_multi_pii-v1 | **71.5%** | 69.2–73.7% | 66.6% | 59.7% | 62.9% | 61.0–64.9% | 55.0% |
| `daslabhq/pii-proxy` (24-label) | **64.1%** | 62.3–65.8% | 83.4% | 58.8% | 69.0% | 67.4–70.5% | 65.8% |
| Presidio (default) | **48.0%** | 46.2–49.8% | 60.6% | 34.0% | 43.6% | 42.0–45.2% | 36.8% |
| regex layer only | **25.6%** | 23.8–27.4% | 72.8% | 20.5% | 32.0% | 29.7–34.2% | 29.6% |
| NVIDIA + regex layer | **91.5%** | 90.2–92.7% | 77.2% | 74.0% | 75.5% | 74.0–77.1% | 72.5% |
| `pii-proxy-ner` + regex layer | **88.8%** | 87.5–90.1% | 79.5% | 71.4% | 75.2% | 73.6–76.8% | 71.6% |
| OpenAI Privacy Filter + regex layer | **80.5%** | 79.2–81.8% | 87.7% | 73.5% | 80.0% | 78.8–81.1% | 76.0% |
| GLiNER2-PII + regex layer | **78.5%** | 76.9–79.8% | 72.6% | 59.9% | 65.6% | 63.8–67.4% | 60.8% |

"+ regex layer" is the model with its full native vocabulary replayed through
pii-proxy's own `detectAll` with the regex layer behind it. GLiNER2-PII drops
there because its native labels have no `time`; the `types` option would ask
for it. Overlap F1 by language: the filter 78–81%, GLiNER2-PII 73–76%, NVIDIA
69–77%, `pii-proxy-ner` 61–75%, knowledgator 55–72%.

Against exp 006: the regex layer reproduces to the decimal (32.0%).
`pii-proxy-ner` + regex 75.2% vs exp 006's 75.7% (the merge is now the
library's own `detectAll` rather than exp 006's emulation of it).

## Long letters (SPY)

[SPY](https://huggingface.co/datasets/mks-logic/SPY): LLM-written legal
questions and medical consultations with Faker values in labelled slots.
Rebuilt from its placeholder files (Faker seed 42), all 8,688 documents;
test split 7,820 documents and 52,911 spans, dev split 868.

| System | masked | exact P | exact R | exact F1 | names as returned | overlap F1 |
|---|---|---|---|---|---|---|
| OpenAI Privacy Filter | **86.7%** | 35.6% | 73.8% | 48.0% | 48.1% | 55.0% |
| `pii-proxy-ner` | 83.8% | 37.4% | 80.2% | 51.0% | 36.6% | 51.8% |
| knowledgator gliner-pii-base | 83.0% | 34.5% | 77.5% | 47.8% | 37.8% | 49.5% |
| NVIDIA gliner-PII | 82.8% | 36.7% | 81.2% | 50.6% | 36.5% | 51.3% |
| GLiNER2-PII | 82.0% | 37.3% | 76.3% | 50.1% | 50.1% | 51.1% |
| gliner_multi_pii-v1 | 77.0% | 26.5% | 69.9% | 38.5% | 38.5% | 39.9% |
| knowledgator, SPY's 7 category names | 87.6% | 36.9% | 82.2% | 50.9% | 51.0% | 52.5% |
| GLiNER2-PII, SPY's 7 category names | 86.1% | 38.5% | 77.3% | 51.4% | 51.5% | 53.7% |
| NVIDIA, SPY's 7 category names | 85.4% | 38.0% | 82.0% | 51.9% | 38.7% | 53.2% |

Per type, the filter masks 71% of SPY's URLs and every other model about 24%
(GLiNER2-PII has no URL label; the others' URL labels rarely fire on bare
personal sites). The filter masks 64% of usernames, the GLiNER models 85–96%.

**SPY's precision measures its labels.** SPY labels only the slots its
generator filled; the LLM-written text around them names more people and
carries more contact details than the slots do: 14,344 unlabeled email
addresses against 8,594 labelled, 9,035 unlabeled US-style phone numbers
against 7,819. A perfect email detector scores 37% precision on SPY. Many of
the 11,644 unlabeled URLs are organisations' sites, which are not personal
data. Of 60 GLiNER2-PII false positives sampled away from any gold span and
any regex-found email, URL or phone, 55 were judged personal data the dataset
left unlabeled (mostly third parties' names) and 5 not; the judgements were
made by an LLM reviewer and wait for a human re-check
([`audit/spy-false-positives.json`](audit/spy-false-positives.json)). Read SPY
for masked share and recall. Thresholds tuned on its dev split learn its gaps
(GLiNER2-PII's EMAIL, NAME and PHONE go to 0.95), so `results.json` keeps them
only to flag them.

### Long documents

| Model | max_len | masked, 200-word windows | masked, whole letter | exact F1, windows | exact F1, whole letter |
|---|---|---|---|---|---|
| NVIDIA gliner-PII | 384 | 82.8% | 54.4% | 50.6% | 50.7% |
| `pii-proxy-ner` | 384 | 83.8% | 55.0% | 51.0% | 50.6% |
| gliner_multi_pii-v1 | 384 | 77.0% | 49.7% | 38.5% | 41.0% |
| knowledgator gliner-pii-base | 2048 | 83.0% | 82.1% | 47.8% | 49.1% |

The GLiNER library cuts every input at the checkpoint's `max_len` (GLiNER
words and punctuation, about 300 plain words on SPY) with one `UserWarning`.
98% of SPY's letters are longer; median 419 words. Exact F1 hides the loss:
recall falls from 81% to 54% while precision rises from 37% to 48%, because
what the cut removes is mostly SPY's unlabeled PII. GLiNER2's library does
not cut (84.7% masked on whole letters with SPY's names, 86.1% in windows).
Every other number in this experiment is on 200-word windows overlapping by 40;
each window owns the text between the midpoints of its overlaps, so each
entity is kept once (`verify.py` checks every SPY document).

### Against the published SPY table

GLiNER2-PII's [report](https://arxiv.org/abs/2605.09973) and model card give
exact-match F1 on 100 legal + 100 medical SPY documents, without the Faker
seed, the document selection, labels, mapping, thresholds or text
reconstruction. Average of legal and medical exact F1:

| Model | ours, windows, native labels | ours, SPY's names | ours, whole letter, SPY's names | card | report | 100+100 draws, 5–95% |
|---|---|---|---|---|---|---|
| GLiNER2-PII | 0.501 | 0.514 | 0.528 | 0.476 | 0.471 | 0.488–0.515 |
| NVIDIA gliner-PII | 0.505 | 0.518 | 0.524 | 0.400 | 0.391 | 0.492–0.518 |
| gliner_multi_pii-v1 | 0.384 | 0.473 | 0.444 | 0.398 | 0.385 | 0.369–0.399 |
| OpenAI Privacy Filter | 0.480 | — | — | 0.380 | 0.373 | 0.466–0.494 |
| knowledgator gliner-pii-base | 0.477 | 0.509 | 0.522 | — | 0.367 | 0.464–0.490 |

None of the published rows reproduces under our scoring. NVIDIA's published
0.39–0.40 matches its score with name parts left unmerged (0.387 with SPY's
names), which suggests the published table counted its split names as misses;
we cannot confirm that. An earlier version of this README attributed the gap
to truncation; truncation barely moves F1 (table above), so that claim is
withdrawn. Our own published numbers do reproduce (Nemotron below; exp 006's
regex layer to the decimal).

## Nemotron-PII held-out

The 500-record `test` set behind the main README's head-to-head (fingerprint
`df099d8d5ff37346`, checked). In-distribution for `pii-proxy-ner` and NVIDIA.

| | 55 native labels, fine F1 | published (main README) |
|---|---|---|
| `pii-proxy-ner` | **92.5%** (P 90.1%, R 94.9%) | 91.5% |
| NVIDIA gliner-PII | **91.8%** (P 90.1%, R 93.6%) | 90.9% |

The difference's 95% interval (paired bootstrap over records) is −0.1 to
+1.3pp: a tie. Both run about 1pp above the published run, which handed whole
texts to the library: 40 of the 500 exceed 384 tokens (2.6% of spans past the
cut). Masked share, coarse view: `pii-proxy-ner` 95.8%, NVIDIA 94.8%,
GLiNER2-PII 88.6%, knowledgator 84.8%, the filter 73.4%.

## Thai

No public Thai PII benchmark was found, so [`build_thai.ts`](build_thai.ts)
makes one: 324 texts (LINE group messages, HR emails, payroll remarks, Thai
mixed with English) from 77 templates, 774 spans: names (414, from the Thai
pack's generators, the title outside the span and written against the name as
Thai is), national IDs with valid check digits (90), phone numbers (126),
emails (54), employee codes (90). No real person is in it. A Thai reader has
not checked it yet.

| System | masked | 95% | names masked | IDs masked | overlap F1 | exact F1 |
|---|---|---|---|---|---|---|
| **GLiNER2-PII + Thai pack + regex, segmented** | **92.0%** | 89.7–94.1% | 90.1% | 88.3% | 96.7% | 86.4% |
| GLiNER2-PII, segmented | 91.5% | 89.3–93.5% | 91.1% | 83.9% | 95.2% | 84.3% |
| OpenAI Privacy Filter, segmented | 70.9% | 67.4–74.5% | 57.0% | 74.4% | 77.1% | 46.9% |
| GLiNER2-PII | 69.4% | 66.1–72.4% | 50.0% | 83.3% | 91.1% | 59.6% |
| OpenAI Privacy Filter | 61.2% | 57.8–64.8% | 40.8% | 70.0% | 72.9% | 51.2% |
| gliner_multi_pii-v1, segmented | 53.5% | 49.7–57.3% | 43.0% | 43.3% | 71.9% | 46.9% |
| GLiNER2-PII, its character splitter | 45.5% | 42.8–48.0% | 10.9% | 70.6% | 60.4% | 58.1% |
| NVIDIA gliner-PII | 42.6% | 39.5–45.4% | 14.0% | 55.0% | 64.3% | 53.8% |
| Thai pack + regex only | 34.9% | 32.7–37.1% | 0.0% | 50.0% | 51.7% | 51.7% |
| `pii-proxy-ner`, segmented | 33.9% | 30.9–36.6% | 11.4% | 54.4% | 44.7% | 43.7% |

GLiNER-family models can only start and end a span on a token of their own
splitter. GLiNER2's splitter keeps Thai vowel and tone marks inside a word, so
an unspaced run (title, name, particle) is one token. GLiNER v1's `\w`
splitter breaks at every mark instead, into pieces that cross word
boundaries: "คุณศิริลักษณ์คะ" splits as `ค|ุ|ณศ|ิ|ร|…`, the title's last
letter glued to the name's first. "Segmented" puts a space between the words
pythainlp's `newmm` finds (only where Thai script is on either side, so IDs,
phones and emails stay whole), runs the model, and maps spans back to the
original characters. GLiNER2's own character-level splitter, meant for
unspaced scripts, does worse than either (names 10.9%).

GLiNER2-PII was fine-tuned on English, French, Spanish, German, Italian,
Portuguese and Dutch; its base model was pretrained on text that includes
Thai. National IDs are found by the Thai pack's check-digit detector
regardless of the model, but the IDs were generated by the same pack.

## Speed, size, licence

| Model | parameters | CPU (2 cores, 4 GB) | labels asked | GPU (A10G) | Licence | From TypeScript |
|---|---|---|---|---|---|---|
| `pii-proxy-ner` | 153M | **1.34 s** | 55 | 38 ms | Apache-2.0 base, ours | GLiNER v1 → ONNX → GLiNER.js |
| knowledgator gliner-pii-base | 166M | 1.52 s | 34 | 42 ms | Apache-2.0 | Official ONNX, GLiNER.js |
| gliner_multi_pii-v1 | 289M | 2.44 s | 36 | 66 ms | Apache-2.0 | Community ONNX, GLiNER.js |
| OpenAI Privacy Filter | 1,399M | 2.56 s | — | 93 ms | Apache-2.0 | Official ONNX, Transformers.js ≥ 4.2 |
| GLiNER2-PII | 307M | 2.58 s | 42 | 71 ms | Apache-2.0 | No official path (one experimental community port) |
| NVIDIA gliner-PII | 445M | 7.08 s | 55 | 112 ms | NVIDIA Open Model License | Community ONNX |
| Presidio | — | 0.13 s | — | (CPU) | MIT | No (Python) |

Median of 30 SPY documents (~450 words), batch size 1, PyTorch fp32, every
model timed in the same 2-core Modal container: containers differ by machine
(NVIDIA's model measured 6.9 s in one and 10.8 s in another), so rows timed
apart are not comparable. Each GLiNER model was asked for its full native
vocabulary (what a pii-proxy detector would ask for). GLiNER reads the labels
together with the text, so they cost time: from 5 labels to all of them,
`pii-proxy-ner` goes from 1.01 s to 1.35 s, GLiNER2-PII from 1.88 s to 2.51 s,
NVIDIA from 5.35 s to 6.93 s ([`latency-labels.json`](latency-labels.json)).
GPU column is throughput per document at batch 16. Parameter counts as loaded
([`facts.json`](facts.json)). Quantized ONNX was not measured.

## Decisions this feeds

What the numbers support; the choices are product calls.

1. **The filter is the strongest identifier detector we measured**, in every
   European language, with the most precise flags, an official ONNX export
   and a Transformers.js runtime. What it leaves (times, states, countries,
   gender) rarely identifies anyone alone. It is 9× the parameters of
   `pii-proxy-ner` and about 2× slower on CPU, and on Thai it trails
   GLiNER2-PII by 20pp. It is the first candidate for the default detector
   that this experiment did not expect.
2. **`pii-proxy-ner`** is the fastest and ties NVIDIA in English, but loses
   5–13pp outside English. Shipping it needs its 55-label checkpoint published
   (the public `daslabhq/pii-proxy` is the 24-label model, 50–64% masked on
   ai4privacy) and an ONNX export.
3. **Thai** wants GLiNER2-PII with the segmenter in the Thai pack. GLiNER2-PII
   has no official ONNX or JS runtime, so that path needs an export first.
4. **Long text must be windowed** in front of any GLiNER v1 checkpoint.
5. **Public pages.** "Matches NVIDIA's flagship" holds on Nemotron and SPY,
   not on ai4privacy (NVIDIA +1.3 to +2.5pp in English, 4.6–6.1pp elsewhere).

## What this benchmark cannot measure

1. **Real documents.** ai4privacy, SPY and Nemotron are synthetic; the Thai
   set is templated by us. None is a real inbox, payroll file or clinical note.
2. **SPY's precision.** Missing labels dominate it.
3. **Thresholds.** Every model runs at its documented default; none was tuned
   on the data scored.
4. **Choices made after seeing results.** Querying with SPY's category names,
   flattening, name merging, the filter's card mapping and the `time`/`date`
   queries were all added after first results; each is reported with and
   without. The regex layer was improved on exp 006's English ai4privacy
   sample, so the regex rows there are in-sample.
5. **Windows.** 200 words with a 40-word overlap, the same for every model. On
   the two models that read long text natively, windows cost about 1.5pp of
   exact F1 and gained about 1pp of masked share.
6. **Label strategy.** One query per model per dataset, not a search.
7. **Quantized and ONNX accuracy.** Everything ran in PyTorch fp32.
8. **Masking quality.** Detection only; fakes and round trips are tested
   elsewhere.

## Setup

| Contender | Weights | Queried with |
|---|---|---|
| GLiNER2-PII | `fastino/gliner2-privacy-filter-PII-multi` (gliner2 2.1.0) | its 42 labels, restricted to what a dataset scores; zero-shot `url`, `time` |
| `pii-proxy-ner` | `modal_train.py --full --full-labels`: gliner_small-v2.1 on Nemotron-PII's 55 labels, 5 epochs, `checkpoint-31220` (the model behind the main README's head-to-head) | native labels |
| `daslabhq/pii-proxy` | the published 24-label model | native labels |
| NVIDIA gliner-PII | `nvidia/gliner-PII` | the same 55 Nemotron labels |
| gliner_multi_pii-v1 | `urchade/gliner_multi_pii-v1` | labels on its card |
| knowledgator | `knowledgator/gliner-pii-base-v1.0` | labels on its card, threshold 0.3 (its card's) |
| OpenAI Privacy Filter | `openai/privacy-filter` | fixed 8 categories, constrained BIOES Viterbi, default operating point |
| Presidio | presidio-analyzer, default recognizers, spaCy `en_core_web_lg` | — |
| regex layer | pii-proxy `defaultDetectors` (+ Thai pack on Thai) | — |

Labels map in two stages ([`mapping.py`](mapping.py)): native label →
canonical type → the dataset's bucket. A detection whose type a dataset does
not annotate is dropped from the typed scores, not counted wrong (exp 006's
rule); masked share counts it. Detections are stored at threshold 0.2 with
scores, so a stricter threshold applied afterwards equals a run at it.
GLiNER2 ends a span that runs to the end of a text one character past it
(100 spans); ends are clamped.

## Reproduce

Heavy work runs on Modal; the machine running these commands only rescores.

```bash
cd experiments/007-detector-benchmark
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python modal faker numpy pythainlp datasets

.venv/bin/python build_data.py               # SPY (seed 42), ai4privacy English (exp 006's), Nemotron
modal run modal_run.py --build-langs         # ai4privacy de/fr/it/es/nl, drawn on Modal
bun build_thai.ts                            # the Thai set (committed: data/thai.jsonl)
modal run modal_run.py                       # every model × dataset on Modal GPUs -> out/
modal run modal_run.py --repro               # whole-letter runs for the published-table comparison -> out-repro/
modal run modal_run.py --latency             # CPU latency, every model in one 2-core container
modal run modal_run.py --model-facts         # parameter counts, checkpoint provenance -> facts.json
modal run modal_run.py --labels-cost         # CPU latency at 5 / 20 / all labels, one container -> latency-labels.json
bun pipeline.ts                              # regex layer, and model + regex through detectAll
.venv/bin/python score.py                    # results.json
.venv/bin/python score.py --out out-repro --json results-repro.json
.venv/bin/python report.py                   # every table above, and the paired intervals
.venv/bin/python verify.py                   # adversarial audit of all of it
```

The `pii-proxy-ner` weights are read from the `pii-proxy-models` Modal volume;
point `PII_PROXY_NER` at a local copy to run it elsewhere.

`verify.py` checks: SPY's per-domain entity counts equal its dataset card;
every gold span has its type's shape; Nemotron's fingerprint and ai4privacy's
counts match exp 005/006; each other language is 1,000 records sharing none
with English; every Thai national ID has a valid check digit; dev and test
share no document; windows partition every document and cut no gold span;
every output covers every record, stays in bounds and was queried with the
labels `mapping.py` gives; gold scores 1.0 against itself and nothing scores
0; detection order does not change a score; a first name alone does not mask
a full name, and name parts merge only where gold labels whole names;
`results.json` re-derives from `out/`; Thai segmentation maps spans back
exactly.

## Files

| File | |
|---|---|
| `build_data.py`, `build_thai.ts` | the evaluation sets |
| `mapping.py` | every label vocabulary and bucket mapping |
| `detectors.py` | one runner per model family, the shared windows, Thai segmentation, BIOES Viterbi |
| `run.py`, `modal_run.py`, `latency.py` | running them locally, on Modal, and timing them |
| `pipeline.ts` | the regex layer, and model + regex through `detectAll` |
| `score.py`, `report.py`, `verify.py` | scoring, the tables, the audit |
| `results.json`, `results-repro.json` | every number above |
| `latency-server-class.json`, `latency-labels.json`, `facts.json` | CPU timings; what labels cost; parameter counts and checkpoint state |
| `audit/spy-false-positives.json` | the 60 judged SPY false positives |
| `data/thai.jsonl` | the Thai set |
