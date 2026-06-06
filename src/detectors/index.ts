export interface Detection {
  type: string;
  value: string;
  start: number;
  end: number;
}

export interface Detector {
  type?: string;
  detect(text: string): Detection[] | Promise<Detection[]>;
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
    const re = /(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}(?!\w)/g;
    const candidates = matchAll(re, text, 'phone');
    return candidates.filter(d => (d.value.match(/\d/g) || []).length >= 7);
  },
};

// ─── Credit Card ────────────────────────────────────────────────

const creditCardDetector: Detector = {
  type: 'credit_card',
  detect(text) {
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
    const v4 = /\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b/g;
    // IPv6: full form, compressed (::), and v4-mapped (::ffff:1.2.3.4).
    // Requires >=2 colon-separated groups so plain times ("12:30") never match.
    const v6 = /(?<![\w:.])(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}|:(?::[0-9a-fA-F]{1,4}){1,7}|::(?:[fF]{4}:)?(?:\d{1,3}\.){3}\d{1,3})(?![\w:.])/g;
    return [...matchAll(v4, text, 'ip_address'), ...matchAll(v6, text, 'ip_address')];
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
    const re = /https?:\/\/[^\s"'<>]+[?&][^\s"'<>]+/g;
    return matchAll(re, text, 'url');
  },
};

// ─── Tracking Number ────────────────────────────────────────────

const trackingDetector: Detector = {
  type: 'tracking_number',
  detect(text) {
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

// ─── Helpers ────────────────────────────────────────────────────

function matchAll(re: RegExp, text: string, type: string): Detection[] {
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

export const defaultDetectors: Detector[] = [
  emailDetector,
  creditCardDetector,
  uuidDetector,
  urlDetector,
  trackingDetector,
  ipDetector,
  phoneDetector,
];

export async function detectAll(text: string, detectors: Detector[] = defaultDetectors): Promise<Detection[]> {
  const allDetections: Detection[] = [];
  for (const detector of detectors) {
    const results = await detector.detect(text);
    allDetections.push(...results);
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
