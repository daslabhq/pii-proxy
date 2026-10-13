import { BijectiveMap } from './map.js';
import { detectAll, defaultDetectors, type Detector, type Detection } from './detectors/index.js';
import { getGenerator, generators, type Generator, type GeneratorContext } from './generators/index.js';
import { LlmDetector, type LlmDetectorOptions } from './detectors/llm.js';

export { BijectiveMap } from './map.js';
export { type Detection, type Detector, defaultDetectors } from './detectors/index.js';
export { type Generator, type GeneratorContext, generators, getGenerator, withheld } from './generators/index.js';
export { LlmDetector, type LlmDetectorOptions } from './detectors/llm.js';
export { knownValues, type KnownValuesOptions } from './detectors/known.js';
export { learn, type LearnOptions, type LearnResult, type LearnedRecord, type LearnedSpan } from './learn.js';

export interface MaskResult {
  text: string;
  detections: Array<Detection & { replacement: string }>;
}

/**
 * A pack bundles detectors and generators for one domain — a locale, an
 * industry, a file format — so a proxy takes it in one line:
 * `new PrivacyProxy({ use: [thai()] })`. A pack's detectors run before the
 * defaults, so its typed IDs win over looser matches on the same span.
 */
export interface Pack {
  detectors?: Detector[];
  generators?: Record<string, Generator>;
  /** Column headers that name a type, for masking tables (`pii-proxy/files`). A null type marks a header as not personal. */
  headers?: Array<[RegExp, string | null]>;
  /** Title prefixes and the sex they imply, so a fake name can follow them. */
  titles?: Array<[RegExp, 'male' | 'female']>;
}

export interface MaskOptions {
  /** Extra detectors for this call, run before the proxy's own. */
  detectors?: Detector[];
  /** Only these types are masked in this call, and only detectors declared for them run. */
  types?: string[];
}

const ENGLISH_TITLES: Array<[RegExp, 'male' | 'female']> = [
  [/^(?:mrs|miss|ms)\b/i, 'female'],
  [/^(?:mr|master)\b/i, 'male'],
];

export interface PrivacyProxyOptions {
  detectors?: Detector[];
  generators?: Record<string, Generator>;
  use?: Pack[];
  seed?: number;
}

/** Real values shorter than this are too common as plain words to police inside a fake. */
const MIN_POLICED_LENGTH = 3;
const MAX_ATTEMPTS = 2000;

export class PrivacyProxy {
  private map: BijectiveMap;
  private detectors: Detector[];
  private gens: Record<string, Generator>;
  private rng: () => number;
  /** Every real value seen or announced: no fake may equal or contain one. */
  private reals = new Set<string>();
  /** Header rules contributed by packs, for table masking. */
  readonly headers: Array<[RegExp, string | null]>;
  private titles: Array<[RegExp, 'male' | 'female']>;

  constructor(options: PrivacyProxyOptions = {}) {
    const packs = options.use ?? [];
    this.map = new BijectiveMap();
    this.detectors = [...packs.flatMap((p) => p.detectors ?? []), ...(options.detectors ?? defaultDetectors)];
    this.gens = Object.assign({}, generators, ...packs.map((p) => p.generators ?? {}), options.generators);
    this.rng = options.seed !== undefined ? seededRng(options.seed) : Math.random;
    this.headers = packs.flatMap((p) => p.headers ?? []);
    this.titles = [...packs.flatMap((p) => p.titles ?? []), ...ENGLISH_TITLES];

    if (options.seed !== undefined) {
      const { faker } = require('@faker-js/faker');
      faker.seed(options.seed);
    }
  }

  static withLocalLlm(
    llmOptions?: LlmDetectorOptions,
    proxyOptions?: Omit<PrivacyProxyOptions, 'detectors'>
  ): PrivacyProxy {
    return new PrivacyProxy({
      ...proxyOptions,
      detectors: [...defaultDetectors, new LlmDetector(llmOptions)],
    });
  }

  async mask(text: string, options: MaskOptions = {}): Promise<MaskResult> {
    const detections = await this.detect(text, options);
    const enriched: MaskResult['detections'] = [];

    let result = text;
    for (let i = detections.length - 1; i >= 0; i--) {
      const d = detections[i];
      const replacement = this.getOrCreateFake(d.value, d.type);
      result = result.slice(0, d.start) + replacement + result.slice(d.end);
      enriched.unshift({ ...d, replacement });
    }

    return { text: result, detections: enriched };
  }

  unmask(text: string): string {
    let result = text;
    const entries = Array.from(this.map.entries())
      .sort((a, b) => b[1].length - a[1].length);

    for (const [real, fake] of entries) {
      let idx = result.indexOf(fake);
      while (idx !== -1) {
        result = result.slice(0, idx) + real + result.slice(idx + fake.length);
        idx = result.indexOf(fake, idx + real.length);
      }
    }
    return result;
  }

  async maskObject<T extends Record<string, unknown>>(obj: T): Promise<{ masked: T; detections: MaskResult['detections'] }> {
    const allDetections: MaskResult['detections'] = [];

    const walk = async (value: unknown): Promise<unknown> => {
      if (typeof value === 'string') {
        const result = await this.mask(value);
        allDetections.push(...result.detections);
        return result.text;
      }
      if (Array.isArray(value)) {
        return Promise.all(value.map(walk));
      }
      if (value && typeof value === 'object') {
        const out: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(value)) {
          out[k] = await walk(v);
        }
        return out;
      }
      return value;
    };

    return { masked: (await walk(obj)) as T, detections: allDetections };
  }

  unmaskObject<T extends Record<string, unknown>>(obj: T): T {
    const walk = (value: unknown): unknown => {
      if (typeof value === 'string') return this.unmask(value);
      if (Array.isArray(value)) return value.map(walk);
      if (value && typeof value === 'object') {
        const out: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(value)) {
          out[k] = walk(v);
        }
        return out;
      }
      return value;
    };
    return walk(obj) as T;
  }

  /** What the proxy's detectors find in a text, overlaps resolved — without masking anything. */
  async detect(text: string, options: MaskOptions = {}): Promise<Detection[]> {
    // With types named, only detectors declared for them run: a model detector stays out of a per-cell check.
    const own = options.types ? this.detectors.filter((d) => d.type && options.types!.includes(d.type)) : this.detectors;
    const found = await detectAll(text, [...(options.detectors ?? []), ...own]);
    return options.types ? found.filter((d) => options.types!.includes(d.type)) : found;
  }

  /** The sex a title implies (Mr, Mrs, and whatever titles the packs add), if it implies one. */
  sexOfTitle(title: string | undefined): 'male' | 'female' | undefined {
    const t = (title ?? '').trim();
    if (!t) return undefined;
    return this.titles.find(([re]) => re.test(t))?.[1];
  }

  getMap(): BijectiveMap {
    return this.map;
  }

  loadMap(data: string): void {
    this.map = BijectiveMap.deserialize(data);
    for (const [real] of this.map.entries()) this.reals.add(real);
  }

  get size(): number {
    return this.map.size;
  }

  /**
   * Announce real values before masking them. Masking a batch — every file
   * of a month, every row of a table — reserves all of it first, so no fake
   * made for the first value can contain a name that only a later one holds.
   */
  reserve(values: Iterable<string>): void {
    for (const v of values) {
      const t = v.trim();
      if (t) this.reals.add(t);
    }
  }

  /** The fake for one real value of a known type: its existing one, or a new one. */
  replace(real: string, type: string, hints: GeneratorContext['hints'] = {}): string {
    return this.getOrCreateFake(real, type, hints);
  }

  /**
   * Record a pairing decided elsewhere — a whole name built from its parts'
   * fakes — so the whole name is found in free text too. Ignored when either
   * side is already taken.
   */
  link(real: string, fake: string, type?: string): void {
    if (!real || !fake || real === fake || this.map.hasFake(real) || this.map.getReal(fake) !== undefined) return;
    this.reals.add(real);
    this.map.set(real, fake, type);
  }

  private getOrCreateFake(real: string, type: string, hints: GeneratorContext['hints'] = {}): string {
    const existing = this.map.getFake(real);
    if (existing) return existing;
    this.reals.add(real.trim());

    const generator = this.gens[type] ?? getGenerator(type);
    const count = this.map.countOf(type);
    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      const fake = generator(real, { attempt, rng: this.rng, hints, count });
      if (this.isFree(fake)) {
        this.map.set(real, fake, type);
        return fake;
      }
    }
    throw new Error(`pii-proxy: no free fake for a ${type} value after ${MAX_ATTEMPTS} tries`);
  }

  /** A fake is free when it is no one's fake yet and neither equals nor contains a real value. */
  private isFree(fake: string): boolean {
    if (this.map.getReal(fake) !== undefined || this.reals.has(fake)) return false;
    const lower = fake.toLowerCase();
    for (const real of this.reals) {
      if (real.length >= MIN_POLICED_LENGTH && lower.includes(real.toLowerCase())) return false;
    }
    return true;
  }
}

/** mulberry32: small, fast and seedable — enough for picking fakes. */
function seededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
