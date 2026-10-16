/**
 * Rows as objects, the shape a database driver or a JSON API hands back:
 * masked the way a table is, and returned in the same shape.
 *
 * The keys become a header row, so columns are found by name exactly as in a
 * spreadsheet. Cells that are not masked come back untouched, in their own
 * type (a number stays a number, a date stays a date). One more pass catches
 * what no header gave away: any remaining text cell equal to a value the map
 * already knows (a manager's name under a column nobody anticipated) gets its
 * fake too.
 */
import type { PrivacyProxy } from '../index.js';
import { maskTable } from './table.js';
import type { Column } from './columns.js';

export interface MaskedRecords {
  records: Record<string, unknown>[];
  columns: Column[];
  people: number;
  /** Cells or spans replaced. */
  masked: number;
}

export async function maskRecords(records: Record<string, unknown>[], proxy: PrivacyProxy): Promise<MaskedRecords> {
  if (records.length === 0) return { records, columns: [], people: 0, masked: 0 };
  const keys = keysOf(records);
  const original = records.map((r) => keys.map((k) => cellText(r[k])));
  const table = await maskTable([keys, ...original], proxy);
  const rows = table.rows.slice(1);
  const known = maskKnownCells(rows, table.columns, proxy);
  return {
    records: records.map((r, i) => restore(r, keys, original[i]!, rows[i]!)),
    columns: table.columns,
    people: table.people,
    masked: table.masked + known,
  };
}

/** Every key any record has, in the order they first appear. */
function keysOf(records: Record<string, unknown>[]): string[] {
  const keys = new Set<string>();
  for (const r of records) for (const k of Object.keys(r)) keys.add(k);
  return [...keys];
}

function cellText(v: unknown): string {
  if (v === null || v === undefined) return '';
  if (v instanceof Date) return Number.isNaN(v.getTime()) ? '' : v.toISOString();
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** Text cells outside the detected columns that equal a value the map holds. */
function maskKnownCells(rows: string[][], columns: Column[], proxy: PrivacyProxy): number {
  const claimed = new Set(columns.filter((c) => c.type !== 'title').map((c) => c.index));
  const map = proxy.getMap();
  let masked = 0;
  for (const row of rows) {
    row.forEach((cell, i) => {
      if (claimed.has(i) || !cell.trim()) return;
      const fake = map.getFake(cell.trim());
      if (fake === undefined) return;
      row[i] = cell.replace(cell.trim(), fake);
      masked++;
    });
  }
  return masked;
}

/** The record with masked cells swapped in, every other cell exactly as it came. */
function restore(record: Record<string, unknown>, keys: string[], before: string[], after: string[]): Record<string, unknown> {
  const out: Record<string, unknown> = { ...record };
  keys.forEach((k, i) => {
    if (after[i] === before[i]) return;
    const value = record[k];
    out[k] = typeof value === 'number' && after[i]!.trim() !== '' && Number.isFinite(Number(after[i])) ? Number(after[i]) : after[i];
  });
  return out;
}
