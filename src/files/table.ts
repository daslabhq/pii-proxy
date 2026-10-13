/**
 * Masking a table: every personal column through the proxy, the header row
 * and every other column left exactly as they were.
 *
 * A full-name column is rebuilt from the same row's masked title, first and
 * last name, in the same layout, so "Mr. John Smith" and its parts never
 * disagree; the pairing is linked in the proxy's map so the whole name is
 * caught in free text too. Cells in no personal column are swept for emails,
 * national IDs and card numbers.
 */
import type { PrivacyProxy } from '../index.js';
import { detectColumns, personalValues, type Column, type ColumnFinding } from './columns.js';

export interface MaskedTable {
  rows: string[][];
  columns: Column[];
  people: number;
  /** Cells or spans replaced. */
  masked: number;
}

const SWEPT_TYPES = ['email', 'national_id', 'credit_card'];

/** Pass `finding` when the columns were detected already — a batch reserves every table's values before masking any. */
export async function maskTable(rows: string[][], proxy: PrivacyProxy, finding?: ColumnFinding): Promise<MaskedTable> {
  const { columns, people } = finding ?? (await detectColumns(rows, proxy));
  proxy.reserve(personalValues(rows, columns));
  const [header = [], ...body] = rows;
  const shape = rowShape(columns);
  let masked = 0;
  const out: string[][] = [];
  for (const row of body) {
    const result = await maskRow(row, columns, shape, proxy);
    masked += result.masked;
    out.push(result.row);
  }
  return { rows: [header, ...out], columns, people, masked };
}

async function maskRow(row: string[], columns: Column[], shape: RowShape, proxy: PrivacyProxy): Promise<{ row: string[]; masked: number }> {
  const next = [...row];
  const sex = proxy.sexOfTitle(shape.titles.map((i) => row[i]).find((t) => t?.trim()));
  let masked = 0;
  // Parts first: a full name is rebuilt from them.
  for (const col of columns) {
    if (col.type === 'title' || col.type === 'person_name') continue;
    const value = row[col.index] ?? '';
    if (!value.trim()) continue;
    next[col.index] = keepEdges(value, proxy.replace(value.trim(), col.type, { sex }));
    masked++;
  }
  for (const col of columns.filter((c) => c.type === 'person_name')) {
    const value = row[col.index] ?? '';
    if (!value.trim()) continue;
    next[col.index] = maskFullName(value, row, next, shape, proxy, sex);
    masked++;
  }
  masked += await sweep(next, columns, proxy);
  return { row: next, masked };
}

// ─── Full names ─────────────────────────────────────────────────

interface RowShape {
  titles: number[];
  firsts: number[];
  lasts: number[];
}

function rowShape(columns: Column[]): RowShape {
  const of = (type: string) => columns.filter((c) => c.type === type).map((c) => c.index);
  return { titles: of('title'), firsts: of('first_name'), lasts: of('last_name') };
}

/**
 * A full name made of this row's title, first and last name is rebuilt from
 * their fakes in the same layout ("Mr." + first + " " + last). Anything else
 * is faked whole, a leading title kept.
 */
function maskFullName(value: string, row: string[], masked: string[], shape: RowShape, proxy: PrivacyProxy, sex: 'male' | 'female' | undefined): string {
  for (const t of [...shape.titles, -1]) {
    const title = t >= 0 ? row[t] ?? '' : '';
    for (const f of shape.firsts) {
      for (const l of shape.lasts) {
        const first = row[f] ?? '';
        const last = row[l] ?? '';
        if (!first.trim()) continue;
        for (const sep of [' ', '  ', '']) {
          for (const glue of ['', ' ']) {
            if (value !== `${title}${glue}${first}${sep}${last}`) continue;
            if (last.trim()) proxy.link(`${first.trim()} ${last.trim()}`, `${masked[f]!.trim()} ${masked[l]!.trim()}`, 'person_name');
            return `${title}${glue}${masked[f]}${sep}${masked[l]}`;
          }
        }
      }
    }
  }
  const title = shape.titles.map((i) => row[i] ?? '').find((t) => t.trim() && value.trimStart().startsWith(t.trim()));
  if (title) {
    const lead = value.slice(0, value.indexOf(title.trim()) + title.trim().length);
    const rest = value.slice(lead.length);
    return lead + keepEdges(rest, proxy.replace(rest.trim(), 'person_name', { sex }));
  }
  return keepEdges(value, proxy.replace(value.trim(), 'person_name', { sex }));
}

// ─── Other cells ────────────────────────────────────────────────

/** Cells in no detected column: emails, national IDs and card numbers are masked wherever they turn up. */
async function sweep(row: string[], columns: Column[], proxy: PrivacyProxy): Promise<number> {
  const claimed = new Set(columns.map((c) => c.index));
  let masked = 0;
  for (const [i, value] of row.entries()) {
    if (claimed.has(i) || !value || value.length < 6) continue;
    const result = await proxy.mask(value, { types: SWEPT_TYPES });
    if (result.detections.length > 0) {
      row[i] = result.text;
      masked += result.detections.length;
    }
  }
  return masked;
}

/** A fake in place of a value, keeping the value's surrounding whitespace. */
function keepEdges(value: string, fake: string): string {
  const m = value.match(/^(\s*)[\s\S]*?(\s*)$/)!;
  return m[1] + fake + m[2];
}
