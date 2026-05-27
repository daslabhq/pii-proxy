import type { Detector, Detection } from './index.js';

export interface LlmDetectorOptions {
  endpoint?: string;
  model?: string;
  entityTypes?: string[];
}

const DEFAULT_ENTITY_TYPES = [
  'person_name',
  'organization',
  'location',
  'date_of_birth',
  'national_id',
  'medical_record',
  'insurance_id',
  'license_plate',
  'passport_number',
];

const SYSTEM_PROMPT = `You are a PII detection engine. Extract all personally identifiable information from text.

Return JSON: {"entities": [{"type": "...", "value": "..."}]}

Example output:
{"entities": [{"type": "person_name", "value": "John Smith"}, {"type": "location", "value": "123 Main St"}]}

Rules:
- "value" must be copied EXACTLY from the input (verbatim, character-for-character)
- Find ALL people — patients, doctors, providers, contacts, references. Include the title (e.g. "Dr. Sarah Chen", not just "Sarah Chen")
- Find ALL instances — multiple people, multiple locations, multiple IDs
- Do NOT include: medical/scientific terms, field labels, public knowledge
- If no PII found, return {"entities": []}`;

function buildUserPrompt(text: string, entityTypes: string[]): string {
  return `Entity types to detect: ${entityTypes.join(', ')}

Text:
"""
${text}
"""`;
}

export class LlmDetector implements Detector {
  private endpoint: string;
  private model: string;
  private entityTypes: string[];

  constructor(options: LlmDetectorOptions = {}) {
    this.endpoint = options.endpoint ?? 'http://localhost:11434/v1/chat/completions';
    this.model = options.model ?? 'qwen3:1.7b';
    this.entityTypes = options.entityTypes ?? DEFAULT_ENTITY_TYPES;
  }

  async detect(text: string): Promise<Detection[]> {
    const body = {
      model: this.model,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: buildUserPrompt(text, this.entityTypes) },
      ],
      temperature: 0,
      response_format: { type: 'json_object' },
    };

    let response: Response;
    try {
      response = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch {
      throw new Error(
        `Cannot reach LLM at ${this.endpoint}. ` +
        `Is Ollama running? Start it with: ollama serve`
      );
    }

    if (!response.ok) {
      const err = await response.text().catch(() => 'unknown error');
      throw new Error(`LLM request failed (${response.status}): ${err}`);
    }

    const data = await response.json() as {
      choices: Array<{ message: { content: string } }>;
    };

    const content = data.choices?.[0]?.message?.content ?? '[]';
    const entities = parseEntities(content);

    return locateEntities(text, entities);
  }
}

interface RawEntity {
  type: string;
  value: string;
}

function parseEntities(content: string): RawEntity[] {
  try {
    const parsed = JSON.parse(content);

    // Standard: array of {type, value}
    const arr = Array.isArray(parsed) ? parsed : parsed.entities ?? parsed.results ?? null;
    if (Array.isArray(arr)) {
      return arr.filter(
        (e: unknown): e is RawEntity =>
          typeof e === 'object' && e !== null &&
          typeof (e as RawEntity).type === 'string' &&
          typeof (e as RawEntity).value === 'string' &&
          (e as RawEntity).value.length > 0
      );
    }

    // Fallback: dict format {"person_name": {"value": "..."}, ...}
    // JSON loses duplicate keys, but we still extract what survived
    if (typeof parsed === 'object' && parsed !== null) {
      const results: RawEntity[] = [];
      for (const [type, val] of Object.entries(parsed)) {
        if (typeof val === 'object' && val !== null && 'value' in val && typeof (val as { value: unknown }).value === 'string') {
          results.push({ type, value: (val as { value: string }).value });
        }
      }
      if (results.length > 0) return results;
    }

    return [];
  } catch {
    return [];
  }
}

function locateEntities(text: string, entities: RawEntity[]): Detection[] {
  const detections: Detection[] = [];

  for (const entity of entities) {
    let searchFrom = 0;
    while (true) {
      const idx = text.indexOf(entity.value, searchFrom);
      if (idx === -1) break;
      detections.push({
        type: entity.type,
        value: entity.value,
        start: idx,
        end: idx + entity.value.length,
      });
      searchFrom = idx + entity.value.length;
    }
  }

  return detections;
}
