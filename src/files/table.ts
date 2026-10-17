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
import { detectColumns, reservePersonal, type Column, type ColumnFinding } from './columns.js';

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
  reservePersonal(rows, columns, proxy);
  const [header = [], ...body] = rows;
  const shape = rowShape(columns);
  const names = columns.filter((c) => c.type === 'person_name');
  let masked = 0;
  // Parts first, in every row, so a full name can be rebuilt from its own row's fakes...
  const out = body.map((row) => {
    const parts = maskParts(row, columns, shape, proxy);
    masked += parts.masked;
    return parts.row;
  });
  // ...then the full names made of those parts, which teaches the map each whole name...
  const rest: Array<[number, Column]> = [];
  body.forEach((row, i) => names.forEach((col) => {
    const value = row[col.index] ?? '';
    if (!value.trim()) return;
    const rebuilt = rebuildFromParts(value, row, out[i]!, shape, proxy);
    if (rebuilt === null) rest.push([i, col]);
    else { out[i]![col.index] = rebuilt; masked++; }
  }));
  // ...and only then the other names, so a manager named on a report's row gets the
  // same fake as on their own row.
  for (const [i, col] of rest) {
    out[i]![col.index] = maskWholeName(body[i]![col.index]!, body[i]!, shape, proxy);
    masked++;
  }
  for (const row of out) masked += await sweep(row, columns, proxy);
  return { rows: [header, ...out], columns, people, masked };
}

/** Every personal column except titles and full names, which are built from these. */
function maskParts(row: string[], columns: Column[], shape: RowShape, proxy: PrivacyProxy): { row: string[]; masked: number } {
  const next = [...row];
  const sex = sexOf(row, shape, proxy);
  let masked = 0;
  for (const col of columns) {
    if (col.type === 'title' || col.type === 'person_name') continue;
    const value = row[col.index] ?? '';
    if (!value.trim()) continue;
    next[col.index] = keepEdges(value, proxy.replace(value.trim(), col.type, { sex }));
    masked++;
  }
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

function sexOf(row: string[], shape: RowShape, proxy: PrivacyProxy): 'male' | 'female' | undefined {
  return proxy.sexOfTitle(shape.titles.map((i) => row[i]).find((t) => t?.trim()));
}

/**
 * A full name made of this row's title, first and last name, rebuilt from
 * their fakes in the same layout ("Mr." + first + " " + last); null when the
 * name is not made of them. The map learns the whole name, with and without
 * its title, so the same person named anywhere else gets the same fake.
 */
function rebuildFromParts(value: string, row: string[], masked: string[], shape: RowShape, proxy: PrivacyProxy): string | null {
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
            const rebuilt = `${title}${glue}${masked[f]}${sep}${masked[l]}`;
            if (last.trim()) proxy.link(`${first.trim()} ${last.trim()}`, `${masked[f]!.trim()} ${masked[l]!.trim()}`, 'person_name');
            proxy.link(value.trim(), rebuilt.trim(), 'person_name');
            return rebuilt;
          }
        }
      }
    }
  }
  return null;
}

/**
 * A name not made of its row's parts: the fake the map already has for it
 * (with or without a leading title), else a new one with the title kept.
 */
function maskWholeName(value: string, row: string[], shape: RowShape, proxy: PrivacyProxy): string {
  const map = proxy.getMap();
  const known = map.getFake(value.trim());
  if (known !== undefined) return keepEdges(value, known);
  const lead = value.match(/^\s*/)![0];
  const body = value.slice(lead.length);
  const title = shape.titles.map((i) => (row[i] ?? '').trim()).find((t) => t && body.startsWith(t)) ?? proxy.titleOf(body);
  if (title) {
    const rest = body.slice(title.length);
    const fake = map.getFake(rest.trim()) ?? proxy.replace(rest.trim(), 'person_name', { sex: proxy.sexOfTitle(title) });
    return lead + title + keepEdges(rest, fake);
  }
  return keepEdges(value, proxy.replace(value.trim(), 'person_name', { sex: sexOf(row, shape, proxy) }));
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
