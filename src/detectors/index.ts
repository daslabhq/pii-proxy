/**
 * PII entity detectors — regex-based for v0.1.
 *
 * Each detector finds entities of a specific type in text and returns
 * their positions. Detectors are ordered by specificity (most specific first)
 * to avoid overlapping matches.
 */

export interface Detection {
  type: EntityType;
  value: string;
  start: number;
  end: number;
}

export type EntityType =
  | 'email'
  | 'phone'
  | 'credit_card'
  | 'ip_address'
  | 'uuid'
  | 'url'
  | 'tracking_number'
  | 'date'
  | 'address';

export interface Detector {
  type: EntityType;
  detect(text: string): Detection[];
}

// ─── Email ──────────────────────────────────────────────────────

const emailDetector: Detector = {
  type: 'email',
  detect(text) {
    const re = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
    return matchAll(re, text, 'email');
  },
};

// ─── Phone ──────────────────────────────────────────────────────

const phoneDetector: Detector = {
  type: 'phone',
  detect(text) {
    // International and common formats: +1-234-567-8901, (234) 567-8901, 234.567.8901
    const re = /(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}(?!\w)/g;
    const candidates = matchAll(re, text, 'phone');
    // Filter: must have at least 7 digits total
    return candidates.filter(d => (d.value.match(/\d/g) || []).length >= 7);
  },
};

// ─── Credit Card ────────────────────────────────────────────────

const creditCardDetector: Detector = {
  type: 'credit_card',
  detect(text) {
    // 13-19 digit numbers, optionally separated by spaces or dashes
    const re = /\b(?:\d[ -]*?){13,19}\b/g;
    const candidates = matchAll(re, text, 'credit_card');
    return candidates.filter(d => luhnCheck(d.value.replace(/\D/g, '')));
  },
};

function luhnCheck(num: string): boolean {
  let sum = 0;
  let alt = false;
  for (let i = num.length - 1; i >= 0; i--) {
    let n = parseInt(num[i], 10);
    if (alt) {
      n *= 2;
      if (n > 9) n -= 9;
    }
    sum += n;
    alt = !alt;
  }
  return sum % 10 === 0;
}

// ─── IP Address ─────────────────────────────────────────────────

const ipDetector: Detector = {
  type: 'ip_address',
  detect(text) {
    // IPv4
    const re = /\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b/g;
    return matchAll(re, text, 'ip_address');
  },
};

// ─── UUID ───────────────────────────────────────────────────────

const uuidDetector: Detector = {
  type: 'uuid',
  detect(text) {
    const re = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;
    return matchAll(re, text, 'uuid');
  },
};

// ─── URL (with tokens/keys in query params) ─────────────────────

const urlDetector: Detector = {
  type: 'url',
  detect(text) {
    // Only match URLs that likely contain sensitive tokens (have query params or long paths)
    const re = /https?:\/\/[^\s"'<>]+[?&][^\s"'<>]+/g;
    return matchAll(re, text, 'url');
  },
};

// ─── Tracking Number ────────────────────────────────────────────

const trackingDetector: Detector = {
  type: 'tracking_number',
  detect(text) {
    // Common carrier formats
    const patterns = [
      /\b1Z[A-Z0-9]{16}\b/g,                           // UPS
      /\b(?:AETH|AP|LEXPU)\d{10,}\w*\b/g,              // AliExpress / Cainiao
      /\b4PX\d{13,}\w*\b/g,                             // 4PX
      /\bTH\d{5}[A-Z0-9]+\b/g,                         // Thailand Post
      /\bLE \d{3} \d{3} \d{3} [A-Z]{2}\b/g,            // Deutsche Post
      /\b(?:92|94|93|95)\d{20,22}\b/g,                  // USPS
      /\b[A-Z]{2}\d{9}[A-Z]{2}\b/g,                    // Universal postal (EMS etc.)
      /\bJD\d{18}\b/g,                                  // Royal Mail
    ];
    const results: Detection[] = [];
    for (const re of patterns) {
      results.push(...matchAll(re, text, 'tracking_number'));
    }
    return results;
  },
};

// ─── Helpers ─────────────────────────────────────────────��──────

function matchAll(re: RegExp, text: string, type: EntityType): Detection[] {
  const results: Detection[] = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    results.push({
      type,
      value: m[0],
      start: m.index,
      end: m.index + m[0].length,
    });
  }
  return results;
}

/** All built-in detectors, ordered by specificity (most specific first). */
export const defaultDetectors: Detector[] = [
  emailDetector,
  creditCardDetector,
  uuidDetector,
  urlDetector,
  trackingDetector,
  ipDetector,
  phoneDetector,
];

/**
 * Run all detectors on a text, removing overlapping detections
 * (earlier/more-specific detectors win).
 */
export function detectAll(text: string, detectors: Detector[] = defaultDetectors): Detection[] {
  const allDetections: Detection[] = [];
  for (const detector of detectors) {
    allDetections.push(...detector.detect(text));
  }

  // Sort by start position, then by length (longer match wins ties)
  allDetections.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));

  // Remove overlaps: earlier detectors win
  const result: Detection[] = [];
  let lastEnd = -1;
  for (const d of allDetections) {
    if (d.start >= lastEnd) {
      result.push(d);
      lastEnd = d.end;
    }
  }

  return result;
}
