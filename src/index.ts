/**
 * pii-proxy — Privacy proxy for AI agents.
 *
 * Mask PII before sending to LLMs, unmask responses to write back to real systems.
 * Uses plausible fake values (not tokens) so LLM reasoning quality is preserved.
 *
 * @example
 * ```ts
 * import { PrivacyProxy } from 'pii-proxy';
 *
 * const proxy = new PrivacyProxy();
 *
 * const masked = proxy.mask("Email alex@example.com about order AETH0000345323DY");
 * // → { text: "Email alex@johnson.net about order BFUI0000482918EZ", detections: [...] }
 *
 * // Send masked.text to your LLM...
 *
 * const real = proxy.unmask("Got it, I'll contact alex@johnson.net");
 * // → "Got it, I'll contact alex@example.com"
 * ```
 */

import { BijectiveMap } from './map.js';
import { detectAll, defaultDetectors, type Detector, type Detection, type EntityType } from './detectors/index.js';
import { generators, type Generator } from './generators/index.js';

export { BijectiveMap } from './map.js';
export { type Detection, type EntityType, type Detector, defaultDetectors } from './detectors/index.js';
export { type Generator, generators } from './generators/index.js';

export interface MaskResult {
  /** The text with PII replaced by plausible fakes. */
  text: string;
  /** All detections found, with their fake replacements. */
  detections: Array<Detection & { replacement: string }>;
}

export interface PrivacyProxyOptions {
  /** Custom detectors to use instead of defaults. */
  detectors?: Detector[];
  /** Custom generators to override defaults for specific entity types. */
  generators?: Partial<Record<EntityType, Generator>>;
  /** Seed for faker to make output deterministic (useful for tests). */
  seed?: number;
}

export class PrivacyProxy {
  private map: BijectiveMap;
  private detectors: Detector[];
  private generators: Record<EntityType, Generator>;

  constructor(options: PrivacyProxyOptions = {}) {
    this.map = new BijectiveMap();
    this.detectors = options.detectors ?? defaultDetectors;
    this.generators = { ...generators, ...options.generators };

    if (options.seed !== undefined) {
      const { faker } = require('@faker-js/faker');
      faker.seed(options.seed);
    }
  }

  /**
   * Mask all detected PII in the text with plausible fake values.
   *
   * The same real value always maps to the same fake value within this proxy
   * instance (deterministic within a session).
   */
  mask(text: string): MaskResult {
    const detections = detectAll(text, this.detectors);
    const enriched: MaskResult['detections'] = [];

    // Build replacements — process from end to start to preserve positions
    let result = text;
    for (let i = detections.length - 1; i >= 0; i--) {
      const d = detections[i];
      const replacement = this.getOrCreateFake(d.value, d.type);
      result = result.slice(0, d.start) + replacement + result.slice(d.end);
      enriched.unshift({ ...d, replacement });
    }

    return { text: result, detections: enriched };
  }

  /**
   * Replace all known fake values in the text with their real originals.
   *
   * Handles the round-trip: mask(text) → send to LLM → unmask(response).
   */
  unmask(text: string): string {
    let result = text;
    // Sort by length descending to avoid partial replacements
    // (e.g., "alex@johnson.net" before "alex")
    const entries = Array.from(this.map.entries())
      .sort((a, b) => b[1].length - a[1].length);

    for (const [real, fake] of entries) {
      // Replace all occurrences of the fake value
      let idx = result.indexOf(fake);
      while (idx !== -1) {
        result = result.slice(0, idx) + real + result.slice(idx + fake.length);
        idx = result.indexOf(fake, idx + real.length);
      }
    }
    return result;
  }

  /**
   * Mask structured data (e.g., tool call input objects).
   * Recursively walks the object and masks all string values.
   */
  maskObject<T extends Record<string, unknown>>(obj: T): { masked: T; detections: MaskResult['detections'] } {
    const allDetections: MaskResult['detections'] = [];

    const walk = (value: unknown): unknown => {
      if (typeof value === 'string') {
        const result = this.mask(value);
        allDetections.push(...result.detections);
        return result.text;
      }
      if (Array.isArray(value)) {
        return value.map(walk);
      }
      if (value && typeof value === 'object') {
        const out: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(value)) {
          out[k] = walk(v);
        }
        return out;
      }
      return value;
    };

    return { masked: walk(obj) as T, detections: allDetections };
  }

  /**
   * Unmask structured data — reverse of maskObject.
   */
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

  /** Get the bijective map (for persistence or debugging). */
  getMap(): BijectiveMap {
    return this.map;
  }

  /** Restore from a previously serialized map. */
  loadMap(data: string): void {
    this.map = BijectiveMap.deserialize(data);
  }

  /** Number of PII entities currently tracked. */
  get size(): number {
    return this.map.size;
  }

  private getOrCreateFake(real: string, type: EntityType): string {
    const existing = this.map.getFake(real);
    if (existing) return existing;

    const generator = this.generators[type];
    let fake = generator(real);

    // Ensure the fake value doesn't collide with another real value
    // or with an existing fake value (unlikely but handle it)
    let attempts = 0;
    while (this.map.getReal(fake) !== undefined && attempts < 10) {
      fake = generator(real);
      attempts++;
    }

    this.map.set(real, fake);
    return fake;
  }
}
