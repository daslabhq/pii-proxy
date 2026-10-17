import type { BijectiveMap } from '../map.js';
import type { Detector, Detection } from './index.js';

/** Types no search looks for alone: a lone first or last name is a word as often as it is a person. */
export const NAME_PARTS = ['first_name', 'last_name'];

const DEFAULT_MIN_LENGTH = 6;
const DEFAULT_MIN_DIGITS = 9;

export interface KnownValuesOptions {
  /** Shorter values are too likely to be ordinary words. Default 6. */
  minLength?: number;
  /** Types never searched for: a lone first name is a word as often as it is a person. */
  exclude?: string[];
  /** A value of digits alone needs this many to be searched for: a shorter one is an amount as often as an ID. Default 9. */
  minDigits?: number;
}

/**
 * Finds the real values a map already holds. A name masked in a table's
 * name column is then caught in a free-text comment two columns over, or in
 * next month's memo, without a model detector.
 */
export function knownValues(map: BijectiveMap, options: KnownValuesOptions = {}): Detector {
  const index = liveIndex(map, options);
  return { detect: (text) => index().find(text) };
}

/**
 * Every real value the map holds, swapped in the text for its fake. Nothing
 * new is made: a value the map has never met stays as it is. Cheap enough to
 * run over everything a system says back (a tool's output, a log line), so
 * a person met once is masked wherever they turn up after.
 */
export function replaceKnown(map: BijectiveMap, options: KnownValuesOptions = {}): (text: string) => string {
  const index = liveIndex(map, options);
  return (text) => {
    const found = index().find(text);
    if (found.length === 0) return text;
    let out = '';
    let at = 0;
    for (const d of found) {
      out += text.slice(at, d.start) + map.getFake(d.value);
      at = d.end;
    }
    return out + text.slice(at);
  };
}

/** The index for the map as it is now; a map only ever grows, so a new size means a rebuild. */
function liveIndex(map: BijectiveMap, options: KnownValuesOptions): () => KnownIndex {
  let built: KnownIndex | null = null;
  let size = -1;
  return () => {
    if (!built || map.size !== size) {
      built = new KnownIndex(map, options);
      size = map.size;
    }
    return built;
  };
}

/**
 * The searchable real values, bucketed by their first few characters, so a
 * text is read once rather than once per value. The leftmost match wins, the
 * longest at that spot, and matches never overlap.
 */
class KnownIndex {
  private readonly buckets = new Map<string, string[]>();
  private readonly keyLength: number;

  constructor(private readonly map: BijectiveMap, options: KnownValuesOptions) {
    const minLength = options.minLength ?? DEFAULT_MIN_LENGTH;
    this.keyLength = Math.max(1, Math.min(minLength, 8));
    for (const [real] of map.entries()) {
      if (!isSearchable(real, map.typeOf(real), options)) continue;
      // A real value that is also someone's fake stays: replacing it would rename a stand-in.
      if (map.getReal(real) !== undefined) continue;
      const key = real.slice(0, this.keyLength);
      const bucket = this.buckets.get(key);
      if (bucket) bucket.push(real);
      else this.buckets.set(key, [real]);
    }
    for (const bucket of this.buckets.values()) bucket.sort((a, b) => b.length - a.length);
  }

  find(text: string): Detection[] {
    const found: Detection[] = [];
    if (this.buckets.size === 0) return found;
    for (let i = 0; i + this.keyLength <= text.length; ) {
      const real = this.matchAt(text, i);
      if (real === undefined) {
        i++;
        continue;
      }
      found.push({ type: this.map.typeOf(real) ?? 'known', value: real, start: i, end: i + real.length });
      i += real.length;
    }
    return found;
  }

  private matchAt(text: string, i: number): string | undefined {
    const bucket = this.buckets.get(text.slice(i, i + this.keyLength));
    return bucket?.find((real) => text.startsWith(real, i) && standsAlone(text, i, i + real.length));
  }
}

/**
 * Whether the search looks for this value at all: long enough not to be an
 * ordinary word, not a lone name part, and not a short run of digits, which
 * is an amount as often as an ID. The proxy polices fakes by the same rule
 * (index.ts), since a value never searched for cannot be mistaken inside one.
 */
export function isSearchable(value: string, type: string | undefined, options: KnownValuesOptions = {}): boolean {
  if (value.length < (options.minLength ?? DEFAULT_MIN_LENGTH)) return false;
  if (type && (options.exclude ?? NAME_PARTS).includes(type)) return false;
  return !(/^\d+$/.test(value) && value.length < (options.minDigits ?? DEFAULT_MIN_DIGITS));
}

const WORD_CHAR = /[A-Za-z0-9]/;

/** A value that starts or ends in a Latin letter or a digit must not run on into another: EMP1234 is not inside EMP12345. */
function standsAlone(text: string, start: number, end: number): boolean {
  const runsIn = start > 0 && WORD_CHAR.test(text[start]!) && WORD_CHAR.test(text[start - 1]!);
  const runsOn = end < text.length && WORD_CHAR.test(text[end - 1]!) && WORD_CHAR.test(text[end]!);
  return !runsIn && !runsOn;
}
