import { BijectiveMap } from './map.js';
import { detectAll, defaultDetectors, type Detector, type Detection } from './detectors/index.js';
import { getGenerator, generators, type Generator } from './generators/index.js';
import { LlmDetector, type LlmDetectorOptions } from './detectors/llm.js';

export { BijectiveMap } from './map.js';
export { type Detection, type Detector, defaultDetectors } from './detectors/index.js';
export { type Generator, generators, getGenerator } from './generators/index.js';
export { LlmDetector, type LlmDetectorOptions } from './detectors/llm.js';

export interface MaskResult {
  text: string;
  detections: Array<Detection & { replacement: string }>;
}

export interface PrivacyProxyOptions {
  detectors?: Detector[];
  generators?: Record<string, Generator>;
  seed?: number;
}

export class PrivacyProxy {
  private map: BijectiveMap;
  private detectors: Detector[];
  private gens: Record<string, Generator>;

  constructor(options: PrivacyProxyOptions = {}) {
    this.map = new BijectiveMap();
    this.detectors = options.detectors ?? defaultDetectors;
    this.gens = { ...generators, ...options.generators };

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

  async mask(text: string): Promise<MaskResult> {
    const detections = await detectAll(text, this.detectors);
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

  getMap(): BijectiveMap {
    return this.map;
  }

  loadMap(data: string): void {
    this.map = BijectiveMap.deserialize(data);
  }

  get size(): number {
    return this.map.size;
  }

  private getOrCreateFake(real: string, type: string): string {
    const existing = this.map.getFake(real);
    if (existing) return existing;

    const generator = this.gens[type] ?? getGenerator(type);
    let fake = generator(real);

    let attempts = 0;
    while (this.map.getReal(fake) !== undefined && attempts < 10) {
      fake = generator(real);
      attempts++;
    }

    this.map.set(real, fake);
    return fake;
  }
}
