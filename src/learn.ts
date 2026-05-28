import { detectAll, defaultDetectors, type Detection } from './detectors/index.js';
import { LlmDetector, type LlmDetectorOptions } from './detectors/llm.js';

export interface LearnOptions {
  llm?: LlmDetectorOptions;
  concurrency?: number;
  onProgress?: (done: number, total: number, current: LearnedRecord) => void;
}

export interface LearnedSpan {
  start: number;
  end: number;
  text: string;
  label: string;
}

export interface LearnedRecord {
  text: string;
  spans: LearnedSpan[];
}

export interface LearnResult {
  records: LearnedRecord[];
  stats: {
    total_records: number;
    total_spans: number;
    entity_counts: Record<string, number>;
    failed_records: number;
  };
}

export async function learn(texts: string[], options: LearnOptions = {}): Promise<LearnResult> {
  const llmDetector = new LlmDetector(options.llm);
  const detectors = [...defaultDetectors, llmDetector];
  const concurrency = options.concurrency ?? 1;

  const records: LearnedRecord[] = [];
  const entityCounts: Record<string, number> = {};
  let failed = 0;

  const process = async (text: string, index: number) => {
    try {
      const detections = await detectAll(text, detectors);

      const spans: LearnedSpan[] = detections.map(d => ({
        start: d.start,
        end: d.end,
        text: d.value,
        label: d.type,
      }));

      const record: LearnedRecord = { text, spans };
      records.push(record);

      for (const s of spans) {
        entityCounts[s.label] = (entityCounts[s.label] ?? 0) + 1;
      }

      options.onProgress?.(records.length, texts.length, record);
    } catch (err) {
      failed++;
      const msg = err instanceof Error ? err.message : String(err);
      options.onProgress?.(records.length, texts.length, { text: `[FAILED: ${msg}]`, spans: [] });
    }
  };

  // Process with concurrency limit
  const queue = texts.map((text, i) => ({ text, index: i }));
  const active: Promise<void>[] = [];

  for (const item of queue) {
    const p = process(item.text, item.index);
    active.push(p);

    if (active.length >= concurrency) {
      await Promise.race(active);
      // Remove resolved promises
      for (let i = active.length - 1; i >= 0; i--) {
        const settled = await Promise.race([active[i].then(() => true), Promise.resolve(false)]);
        if (settled) active.splice(i, 1);
      }
    }
  }

  await Promise.all(active);

  return {
    records,
    stats: {
      total_records: records.length,
      total_spans: records.reduce((sum, r) => sum + r.spans.length, 0),
      entity_counts: entityCounts,
      failed_records: failed,
    },
  };
}

export function learnResultToTrainingData(result: LearnResult): string {
  return result.records.map(r => JSON.stringify(r)).join('\n');
}
