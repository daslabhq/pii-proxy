/**
 * pii-proxy/files — mask whole files: spreadsheets and CSV column by column,
 * text through the proxy's detectors.
 *
 *   import { PrivacyProxy } from 'pii-proxy';
 *   import { maskFiles } from 'pii-proxy/files';
 *
 *   const proxy = new PrivacyProxy();   // packs (pii-proxy/packs/*) add their locale's headers and names
 *   const { results } = await maskFiles([{ name: 'payroll.xlsx', bytes }], proxy);
 *   // results[0].files[0] → { name: 'payroll (masked).csv', mimeType: 'text/csv', bytes }
 *
 * Every file goes through the same proxy, so the same person gets the same
 * fake in every file of the batch, and in every later batch that loads the
 * proxy's map. A workbook comes back as one CSV per sheet: the masked copy is
 * for reading, and CSV is what every tool, agent and spreadsheet opens.
 */
import type { PrivacyProxy } from '../index.js';
import { knownValues } from '../detectors/known.js';
import { detectColumns, personalValues, type Column, type ColumnFinding } from './columns.js';
import { maskTable } from './table.js';
import { parseCsv, toCsv } from './csv.js';
import { readXlsx, type Sheet } from './xlsx.js';

export { detectColumns, type Column, type ColumnFinding } from './columns.js';
export { maskTable, type MaskedTable } from './table.js';
export { parseCsv, toCsv } from './csv.js';
export { readXlsx, excelDate, type Sheet } from './xlsx.js';

export interface InputFile {
  name: string;
  bytes: Uint8Array;
  mimeType?: string;
}

export interface MaskedFile {
  name: string;
  mimeType: string;
  bytes: Uint8Array;
}

export interface FileResult {
  source: string;
  files: MaskedFile[];
  /** Distinct people the file names (its largest table's count). */
  people: number;
  /** Cells or spans replaced. */
  masked: number;
  /** The columns masked, by header — so a reader can check what was and was not touched. */
  columns: Array<{ header: string; type: string }>;
}

export interface Skipped {
  source: string;
  reason: string;
}

/**
 * Mask files that belong together — one month's payroll and its bank file —
 * in one call. Every personal value of the batch is reserved before any fake
 * is made, so no fake made for the first file contains a name the last holds.
 */
export async function maskFiles(inputs: InputFile[], proxy: PrivacyProxy): Promise<{ results: FileResult[]; skipped: Skipped[] }> {
  const { read, skipped } = readAll(inputs);
  const findings = await findColumns(read, proxy);
  for (const [i, f] of read.entries()) {
    if (f.kind === 'sheets') f.sheets.forEach((s, j) => proxy.reserve(personalValues(s.rows, findings[i]![j]!.columns)));
  }
  const results: FileResult[] = [];
  for (const [i, f] of read.entries()) {
    results.push(f.kind === 'sheets' ? await maskSheets(f, findings[i]!, proxy) : await maskTextFile(f, proxy));
  }
  return { results, skipped };
}

/** Free text: values the map already holds (whole names, codes), then everything the proxy's detectors find. */
export async function maskText(text: string, proxy: PrivacyProxy): Promise<{ text: string; masked: number }> {
  const result = await proxy.mask(text, { detectors: [knownValues(proxy.getMap())] });
  return { text: result.text, masked: result.detections.length };
}

export type FileKind = 'workbook' | 'table' | 'text';

/** What a file is, by name and type — null when it cannot be masked yet (PDF, images). */
export function fileKind(name: string, mimeType = ''): FileKind | null {
  const ext = name.toLowerCase().match(/\.([a-z0-9]+)$/)?.[1] ?? '';
  if (ext === 'xlsx' || mimeType.includes('spreadsheetml')) return 'workbook';
  if (ext === 'csv' || ext === 'tsv' || mimeType === 'text/csv' || mimeType === 'text/tab-separated-values') return 'table';
  if (['txt', 'md', 'markdown', 'json', 'log'].includes(ext) || mimeType.startsWith('text/')) return 'text';
  return null;
}

// ─── Reading ────────────────────────────────────────────────────

type ReadFile =
  | { kind: 'sheets'; name: string; sheets: Sheet[] }
  | { kind: 'text'; name: string; mimeType: string; text: string };

function readAll(inputs: InputFile[]): { read: ReadFile[]; skipped: Skipped[] } {
  const read: ReadFile[] = [];
  const skipped: Skipped[] = [];
  for (const input of inputs) {
    const kind = fileKind(input.name, input.mimeType);
    if (!kind) {
      skipped.push({ source: input.name, reason: 'only .xlsx, .csv and text files can be masked' });
      continue;
    }
    try {
      read.push(readOne(input, kind));
    } catch {
      // Never the parser's message: it can quote the cell it choked on.
      skipped.push({ source: input.name, reason: 'could not be read — is it a valid spreadsheet?' });
    }
  }
  return { read, skipped };
}

function readOne(input: InputFile, kind: FileKind): ReadFile {
  if (kind === 'workbook') return { kind: 'sheets', name: input.name, sheets: readXlsx(input.bytes) };
  const text = new TextDecoder('utf-8').decode(input.bytes);
  if (kind === 'table') return { kind: 'sheets', name: input.name, sheets: [{ name: '', rows: parseCsv(text) }] };
  return { kind: 'text', name: input.name, mimeType: input.mimeType || 'text/plain', text };
}

async function findColumns(read: ReadFile[], proxy: PrivacyProxy): Promise<ColumnFinding[][]> {
  const findings: ColumnFinding[][] = [];
  for (const f of read) {
    findings.push(f.kind === 'sheets' ? await Promise.all(f.sheets.map((s) => detectColumns(s.rows, proxy))) : []);
  }
  return findings;
}

// ─── Masking ────────────────────────────────────────────────────

async function maskSheets(f: Extract<ReadFile, { kind: 'sheets' }>, findings: ColumnFinding[], proxy: PrivacyProxy): Promise<FileResult> {
  const base = f.name.replace(/\.[^.]+$/, '');
  const result: FileResult = { source: f.name, files: [], people: 0, masked: 0, columns: [] };
  for (const [j, sheet] of f.sheets.entries()) {
    const table = await maskTable(sheet.rows, proxy, findings[j]);
    result.people = Math.max(result.people, table.people);
    result.masked += table.masked;
    result.columns.push(...masked(table.columns));
    const name = f.sheets.length > 1 ? `${base} - ${sheet.name} (masked).csv` : `${base} (masked).csv`;
    result.files.push({ name, mimeType: 'text/csv', bytes: new TextEncoder().encode(toCsv(table.rows)) });
  }
  return result;
}

async function maskTextFile(f: Extract<ReadFile, { kind: 'text' }>, proxy: PrivacyProxy): Promise<FileResult> {
  const { text, masked } = await maskText(f.text, proxy);
  const name = f.name.replace(/(\.[^.]+)?$/, (ext) => ` (masked)${ext}`);
  return { source: f.name, files: [{ name, mimeType: f.mimeType, bytes: new TextEncoder().encode(text) }], people: 0, masked, columns: [] };
}

function masked(columns: Column[]): Array<{ header: string; type: string }> {
  return columns.filter((c) => c.type !== 'title').map((c) => ({ header: c.header, type: c.type }));
}
