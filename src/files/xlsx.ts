/**
 * The sheets of an .xlsx as text cells, read straight from the zip: shared
 * and inline strings, numbers at the shortest form that round-trips (what
 * Excel shows), date-styled cells as ISO dates.
 */
import { unzipSync, strFromU8 } from "fflate";

export interface Sheet {
  name: string;
  rows: string[][];
}

/** Every sheet of a workbook as text cells. Dates become ISO dates, the rest stays as stored. */
export function readXlsx(bytes: Uint8Array): Sheet[] {
  const zip = unzipSync(bytes);
  const read = (path: string) => (zip[path] ? strFromU8(zip[path]!) : null);
  const workbook = read("xl/workbook.xml");
  if (!workbook) throw new Error("not an .xlsx workbook");
  const shared = sharedStrings(read("xl/sharedStrings.xml"));
  const dateStyles = dateStyleIndexes(read("xl/styles.xml"));
  const date1904 = /<workbookPr[^>]*date1904="(1|true)"/.test(workbook);
  return sheetsOf(workbook, read("xl/_rels/workbook.xml.rels")).map(({ name, target }) => ({
    name,
    rows: sheetRows(read(`xl/${target.replace(/^\/?xl\//, "")}`) ?? "", shared, dateStyles, date1904),
  }));
}

function sheetsOf(workbook: string, rels: string | null): Array<{ name: string; target: string }> {
  const targets = new Map<string, string>();
  for (const m of (rels ?? "").matchAll(/<Relationship\b([^>]*)\/?>/g)) {
    const id = attr(m[1]!, "Id");
    const target = attr(m[1]!, "Target");
    if (id && target) targets.set(id, target);
  }
  const sheets: Array<{ name: string; target: string }> = [];
  for (const m of workbook.matchAll(/<sheet\b([^>]*)\/?>/g)) {
    const target = targets.get(attr(m[1]!, "r:id") ?? "");
    if (target) sheets.push({ name: decodeXml(attr(m[1]!, "name") ?? "Sheet"), target });
  }
  return sheets;
}

function sharedStrings(xml: string | null): string[] {
  if (!xml) return [];
  return [...xml.matchAll(/<si\b[^>]*>([\s\S]*?)<\/si>/g)].map((m) => textRuns(m[1]!));
}

/** The text of an <si> or <is>: its <t> runs joined, phonetic runs (<rPh>) left out. */
function textRuns(xml: string): string {
  const withoutPhonetic = xml.replace(/<rPh\b[\s\S]*?<\/rPh>/g, "");
  return [...withoutPhonetic.matchAll(/<t\b[^>]*>([\s\S]*?)<\/t>/g)].map((m) => decodeXml(m[1]!)).join("");
}

const BUILTIN_DATE_FORMATS = new Set([14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 30, 36, 45, 46, 47, 50, 57]);

/** Indexes into cellXfs whose number format shows a date or a time. */
function dateStyleIndexes(xml: string | null): Set<number> {
  if (!xml) return new Set();
  const custom = new Map<number, string>();
  for (const m of xml.matchAll(/<numFmt\b([^>]*)\/?>/g)) {
    const id = Number(attr(m[1]!, "numFmtId"));
    custom.set(id, decodeXml(attr(m[1]!, "formatCode") ?? ""));
  }
  const xfs = xml.match(/<cellXfs\b[^>]*>([\s\S]*?)<\/cellXfs>/)?.[1] ?? "";
  const dates = new Set<number>();
  [...xfs.matchAll(/<xf\b([^>]*?)(\/>|>)/g)].forEach((m, index) => {
    const id = Number(attr(m[1]!, "numFmtId") ?? 0);
    if (BUILTIN_DATE_FORMATS.has(id) || isDateFormat(custom.get(id))) dates.add(index);
  });
  return dates;
}

function isDateFormat(code: string | undefined): boolean {
  if (!code) return false;
  const bare = code.replace(/"[^"]*"|\[[^\]]*\]|\\./g, "");
  return /[dmyhs]/i.test(bare) && !/^general$/i.test(bare.trim());
}

function sheetRows(xml: string, shared: string[], dateStyles: Set<number>, date1904: boolean): string[][] {
  const rows: string[][] = [];
  for (const rowMatch of xml.matchAll(/<row\b([^>]*)>([\s\S]*?)<\/row>/g)) {
    const rowIndex = Number(attr(rowMatch[1]!, "r") ?? rows.length + 1) - 1;
    const cells: string[] = [];
    for (const c of rowMatch[2]!.matchAll(/<c\b([^>]*?)(?:\/>|>([\s\S]*?)<\/c>)/g)) {
      const attrs = c[1]!;
      const ref = attr(attrs, "r");
      const col = ref ? columnIndex(ref) : cells.length;
      cells[col] = cellText(attrs, c[2] ?? "", shared, dateStyles, date1904);
    }
    rows[rowIndex] = Array.from({ length: cells.length }, (_, i) => cells[i] ?? "");
  }
  const width = Math.max(0, ...rows.map((r) => r?.length ?? 0));
  return Array.from({ length: rows.length }, (_, i) => {
    const r = rows[i] ?? [];
    return Array.from({ length: width }, (_, j) => r[j] ?? "");
  });
}

function cellText(attrs: string, inner: string, shared: string[], dateStyles: Set<number>, date1904: boolean): string {
  const type = attr(attrs, "t");
  if (type === "inlineStr") return textRuns(inner.match(/<is\b[^>]*>([\s\S]*?)<\/is>/)?.[1] ?? "");
  const v = inner.match(/<v\b[^>]*>([\s\S]*?)<\/v>/)?.[1];
  if (v == null) return "";
  if (type === "s") return shared[Number(v)] ?? "";
  if (type === "b") return v === "1" ? "TRUE" : "FALSE";
  if (type === "str" || type === "e") return decodeXml(v);
  const style = Number(attr(attrs, "s") ?? -1);
  if (dateStyles.has(style) && /^-?\d+(\.\d+)?$/.test(v)) return excelDate(Number(v), date1904);
  // Workbooks store doubles at full precision (10231.719999999999); the
  // shortest form that round-trips is what Excel shows (10231.72).
  const n = Number(v);
  return v.trim() !== "" && Number.isFinite(n) ? String(n) : v;
}

/** An Excel serial date as ISO: a date alone when there is no time of day. */
export function excelDate(serial: number, date1904 = false): string {
  const epoch = date1904 ? Date.UTC(1904, 0, 1) : Date.UTC(1899, 11, 30);
  const ms = epoch + Math.round(serial * 86_400_000);
  const iso = new Date(ms).toISOString();
  return Number.isInteger(serial) ? iso.slice(0, 10) : iso.slice(0, 19).replace("T", " ");
}

function columnIndex(ref: string): number {
  const letters = ref.match(/^[A-Z]+/)?.[0] ?? "A";
  return [...letters].reduce((n, ch) => n * 26 + (ch.charCodeAt(0) - 64), 0) - 1;
}

function attr(attrs: string, name: string): string | undefined {
  const m = attrs.match(new RegExp(`(?:^|\\s)${name.replace(":", "\\:")}="([^"]*)"`));
  return m?.[1];
}

function decodeXml(s: string): string {
  return s.replace(/&(lt|gt|amp|quot|apos|#x[0-9a-fA-F]+|#\d+);/g, (_, e: string) => {
    if (e === "lt") return "<";
    if (e === "gt") return ">";
    if (e === "amp") return "&";
    if (e === "quot") return '"';
    if (e === "apos") return "'";
    return String.fromCodePoint(e.startsWith("#x") ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10));
  });
}
