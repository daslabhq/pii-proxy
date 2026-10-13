import type { BijectiveMap } from '../map.js';
import type { Detector, Detection } from './index.js';

export interface KnownValuesOptions {
  /** Shorter values are too likely to be ordinary words. Default 6. */
  minLength?: number;
  /** Types never searched for: a lone first name is a word as often as it is a person. */
  exclude?: string[];
}

/**
 * Finds the real values a map already holds. A name masked in a table's
 * name column is then caught in a free-text comment two columns over, or in
 * next month's memo, without a model detector.
 */
export function knownValues(map: BijectiveMap, options: KnownValuesOptions = {}): Detector {
  const minLength = options.minLength ?? 6;
  const exclude = new Set(options.exclude ?? ['first_name', 'last_name']);
  return {
    detect(text) {
      const found: Detection[] = [];
      const reals = Array.from(map.entries(), ([real]) => real)
        .filter((real) => real.length >= minLength && !exclude.has(map.typeOf(real) ?? ''))
        .sort((a, b) => b.length - a.length);
      for (const real of reals) {
        for (let i = text.indexOf(real); i >= 0; i = text.indexOf(real, i + real.length)) {
          found.push({ type: map.typeOf(real) ?? 'known', value: real, start: i, end: i + real.length });
        }
      }
      return found;
    },
  };
}
