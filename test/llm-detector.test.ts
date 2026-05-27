import { describe, test, expect, mock } from 'bun:test';
import { LlmDetector } from '../src/detectors/llm.js';

function mockFetch(entities: Array<{ type: string; value: string }>) {
  return mock(() =>
    Promise.resolve(new Response(JSON.stringify({
      choices: [{ message: { content: JSON.stringify(entities) } }],
    })))
  );
}

describe('LlmDetector', () => {
  test('detects person names with correct positions', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch([
      { type: 'person_name', value: 'Marcus Weber' },
    ]) as typeof fetch;

    try {
      const detector = new LlmDetector();
      const detections = await detector.detect('Patient: Marcus Weber has an appointment');

      expect(detections).toHaveLength(1);
      expect(detections[0].type).toBe('person_name');
      expect(detections[0].value).toBe('Marcus Weber');
      expect(detections[0].start).toBe(9);
      expect(detections[0].end).toBe(21);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('finds all occurrences of same entity', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch([
      { type: 'person_name', value: 'Sarah Chen' },
    ]) as typeof fetch;

    try {
      const detector = new LlmDetector();
      const text = 'Dr. Sarah Chen referred to Sarah Chen notes';
      const detections = await detector.detect(text);

      expect(detections).toHaveLength(2);
      expect(detections[0].start).toBe(4);
      expect(detections[1].start).toBe(27);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('skips entities not found in text', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch([
      { type: 'person_name', value: 'Hallucinated Name' },
    ]) as typeof fetch;

    try {
      const detector = new LlmDetector();
      const detections = await detector.detect('No names here');

      expect(detections).toHaveLength(0);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('handles multiple entity types', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch([
      { type: 'person_name', value: 'Marcus Weber' },
      { type: 'organization', value: 'Heidelberg University' },
      { type: 'location', value: 'Mannheim' },
    ]) as typeof fetch;

    try {
      const detector = new LlmDetector();
      const text = 'Marcus Weber works at Heidelberg University in Mannheim';
      const detections = await detector.detect(text);

      expect(detections).toHaveLength(3);
      expect(detections.map(d => d.type)).toEqual([
        'person_name', 'organization', 'location'
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('handles LLM returning wrapped JSON', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mock(() =>
      Promise.resolve(new Response(JSON.stringify({
        choices: [{
          message: {
            content: JSON.stringify({
              entities: [{ type: 'person_name', value: 'Test User' }]
            })
          }
        }],
      })))
    ) as typeof fetch;

    try {
      const detector = new LlmDetector();
      const detections = await detector.detect('Hello Test User');

      expect(detections).toHaveLength(1);
      expect(detections[0].value).toBe('Test User');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('handles empty LLM response', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch([]) as typeof fetch;

    try {
      const detector = new LlmDetector();
      const detections = await detector.detect('No PII here');
      expect(detections).toHaveLength(0);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
