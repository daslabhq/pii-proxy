/**
 * Which columns of a table hold personal data, and how many people it names.
 *
 * Column by column: the header first — the English vocabulary here plus
 * whatever headers the proxy's packs add — then the values when the header
 * says nothing. A column's type is a proxy type (`first_name`, `employee_id`,
 * `national_id`, …), so its values are faked by the proxy's generator for
 * that type. A `title` column is read, not masked: it picks a fake name's sex.
 */
import type { PrivacyProxy } from '../index.js';

export interface Column {
  index: number;
  header: string;
  type: string;
}

export interface ColumnFinding {
  columns: Column[];
  /** Distinct people the table names: by code, else by full name, else by first + last. */
  people: number;
}

export async function detectColumns(rows: string[][], proxy: PrivacyProxy): Promise<ColumnFinding> {
  const [header = [], ...body] = rows;
  const columns: Column[] = [];
  for (const [index, name] of header.entries()) {
    const values = body.map((r) => r[index] ?? '');
    const type = await typeOf(name, values, proxy);
    if (type) columns.push({ index, header: name, type });
  }
  return { columns, people: countPeople(body, columns) };
}

/** The values a table's personal columns hold — what a batch reserves before faking anything. */
/** Every personal value of a table, announced to the proxy with its column's type before any fake is made. */
export function reservePersonal(rows: string[][], columns: Column[], proxy: PrivacyProxy): void {
  for (const c of columns) {
    if (c.type === 'title') continue;
    proxy.reserve(rows.slice(1).map((row) => (row[c.index] ?? '').trim()).filter(Boolean), c.type);
  }
}

async function typeOf(header: string, values: string[], proxy: PrivacyProxy): Promise<string | null> {
  const fromHeader = typeFromHeader(header, proxy.headers);
  if (fromHeader === null) return null;
  if (fromHeader === undefined) return typeFromValues(values, proxy);
  return (await confirmedByValues(fromHeader, values, proxy)) ?? typeFromValues(values, proxy);
}

// ─── By header ──────────────────────────────────────────────────

const PERSON_PREFIX = '(?:emp|employee|staff|person|personnel|worker|customer|contact|member|patient|client|user|guest|student|full|display|real|legal)?';
const LANG_SUFFIX = '(?:eng|en|th|local)?';

/** Who anyone is, in any field: names, titles, IDs and contact details. Matched against the squashed header. */
const ENGLISH: Array<[RegExp, string]> = [
  [new RegExp(`^${PERSON_PREFIX}(?:firstname|givenname|forename|fname)${LANG_SUFFIX}$`), 'first_name'],
  [new RegExp(`^${PERSON_PREFIX}(?:lastname|surname|familyname|lname)${LANG_SUFFIX}$`), 'last_name'],
  [new RegExp(`^${PERSON_PREFIX}(?:middlename|mname|nickname)${LANG_SUFFIX}$`), 'first_name'],
  [new RegExp(`^${PERSON_PREFIX}(?:birthdate|dateofbirth|dob|birthday)$`), 'date_of_birth'],
  [new RegExp(`^${PERSON_PREFIX}name${LANG_SUFFIX}$`), 'person_name'],
  [new RegExp(`^${PERSON_PREFIX}fullname${LANG_SUFFIX}$`), 'person_name'],
  [new RegExp(`^(?:emp|employee|person)?(?:title|prefix|salutation)${LANG_SUFFIX}$`), 'title'],
  [/(?:nationalid|citizenid|identitycard|identityno|idcard|idcardno|idno|passport(?:no|number)?|taxid|ssn|socialsecurity(?:no|number|id))/, 'national_id'],
  [/^(?:e?mail|emailaddress|.*email)$/, 'email'],
  [/(?:phone|mobile|telephone|^tel(?:no)?$|^fax)/, 'phone'],
  [/(?:address|street|^addr)/, 'address'],
  [/(?:bankaccount|accountno|accountnumber|^iban$|bankacc)/, 'bank_account'],
];

/**
 * A type, null when a rule says the header is not personal, undefined when
 * nothing matched. Pack rules come first and see the header both as written
 * (an item number like "14." dropped) and squashed like the core's own.
 */
function typeFromHeader(header: string, packRules: Array<[RegExp, string | null]>): string | null | undefined {
  const plain = header.trim().replace(/^\d+\.\s*/, '');
  const word = squash(header);
  for (const [re, type] of packRules) if (re.test(plain) || (word && re.test(word))) return type;
  if (!word) return undefined;
  return ENGLISH.find(([re]) => re.test(word))?.[1];
}

/** "Emp First Name" → "empfirstname". */
function squash(header: string): string {
  return header.toLowerCase().replace(/[\s_\-.()/]+/g, '');
}

// ─── By values ──────────────────────────────────────────────────

const VALUE_TYPES = ['email', 'national_id', 'phone'];
const SAMPLE = 200;

/** A header that promises contact details must hold some: a "phone" column of zeros is an amount. Names and codes are trusted from the header. */
async function confirmedByValues(type: string, values: string[], proxy: PrivacyProxy): Promise<string | null> {
  if (!VALUE_TYPES.includes(type)) return type;
  const filled = sample(values.filter((v) => v !== '0'));
  if (filled.length === 0) return null;
  return (await share(filled, type, proxy)) >= 0.3 ? type : null;
}

/** A column whose header says nothing, judged by what it holds: most values must agree. */
async function typeFromValues(values: string[], proxy: PrivacyProxy): Promise<string | null> {
  const filled = sample(values);
  if (filled.length < 3) return null;
  for (const type of VALUE_TYPES) if ((await share(filled, type, proxy)) >= 0.6) return type;
  return null;
}

/** The share of values that are, whole, one value of this type by the proxy's own detectors. */
async function share(values: string[], type: string, proxy: PrivacyProxy): Promise<number> {
  let hits = 0;
  for (const v of values) {
    const found = await proxy.detect(v, { types: [type] });
    if (found.some((d) => d.value.length >= v.length - 1)) hits++;
  }
  return hits / values.length;
}

function sample(values: string[]): string[] {
  return values.map((v) => v.trim()).filter(Boolean).slice(0, SAMPLE);
}

// ─── Counting people ────────────────────────────────────────────

function countPeople(body: string[][], columns: Column[]): number {
  const key = keyColumns(columns);
  if (key.length === 0) return 0;
  const people = new Set<string>();
  for (const row of body) {
    const id = key.map((c) => (row[c.index] ?? '').trim()).join('\u0000');
    if (id.replace(/\u0000/g, '')) people.add(id);
  }
  return people.size;
}

/** What identifies a person here, most specific first. */
function keyColumns(columns: Column[]): Column[] {
  const first = (type: string) => columns.find((c) => c.type === type);
  const code = first('employee_id') ?? first('national_id');
  if (code) return [code];
  const full = first('person_name');
  if (full) return [full];
  return [first('first_name'), first('last_name')].filter((c): c is Column => !!c);
}
