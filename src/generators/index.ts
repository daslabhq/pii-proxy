import { faker } from '@faker-js/faker';

/** What a generator may use beyond the real value. Generators that need none of it ignore it. */
export interface GeneratorContext {
  /** 0 on the first try; higher when an earlier fake was taken, so a generator can widen its range. */
  attempt: number;
  /** Uniform in [0, 1). Seeded when the proxy is. */
  rng: () => number;
  /** What the surrounding record says about the value, e.g. the sex a title implies. */
  hints: { sex?: 'male' | 'female' };
  /** How many values of this type the map already holds — for numbered stand-ins. */
  count: number;
}

export type Generator = (real: string, ctx: GeneratorContext) => string;

// Reserved example domains: a fake address can never deliver to a stranger.
const emailGenerator: Generator = () => {
  return faker.internet.exampleEmail().toLowerCase();
};

const phoneGenerator: Generator = (real) => {
  const hasPlus = real.startsWith('+');
  const phone = faker.phone.number();
  if (hasPlus && !phone.startsWith('+')) return '+' + phone;
  return phone;
};

const creditCardGenerator: Generator = (real) => {
  const separator = real.includes('-') ? '-' : real.includes(' ') ? ' ' : '';
  const fake = faker.finance.creditCardNumber();
  if (!separator) return fake.replace(/\D/g, '');
  const groups = real.split(/[-\s]/);
  const fakeClean = fake.replace(/\D/g, '');
  let pos = 0;
  return groups.map(g => {
    const chunk = fakeClean.slice(pos, pos + g.length);
    pos += g.length;
    return chunk;
  }).join(separator);
};

const ipGenerator: Generator = (real) => {
  // Match the address family so the fake is shaped like the real value.
  return real.includes(':') ? faker.internet.ipv6() : faker.internet.ipv4();
};

const uuidGenerator: Generator = () => {
  return faker.string.uuid();
};

const urlGenerator: Generator = () => {
  return `https://example.com/${faker.string.alphanumeric(8)}?token=${faker.string.alphanumeric(16)}`;
};

const trackingGenerator: Generator = (real) => {
  return real.replace(/[A-Z]/g, () => faker.string.alpha({ length: 1, casing: 'upper' }))
             .replace(/[0-9]/g, () => String(faker.number.int({ min: 0, max: 9 })));
};

const firstNameGenerator: Generator = (_real, ctx) => {
  const first = faker.person.firstName(ctx?.hints?.sex);
  // About 500 per sex: once a large staff has used them up, double names, as Mary-Ann.
  return (ctx?.attempt ?? 0) < 20 ? first : `${first}-${faker.person.firstName(ctx?.hints?.sex)}`;
};

const lastNameGenerator: Generator = (_real, ctx) => {
  const last = faker.person.lastName();
  // About ten thousand in all: a staff larger than that gets double-barrelled ones.
  return (ctx?.attempt ?? 0) < 20 ? last : `${last}-${faker.person.lastName()}`;
};

const personNameGenerator: Generator = (_real, ctx) => {
  return `${faker.person.firstName(ctx?.hints?.sex)} ${faker.person.lastName()}`;
};

const organizationGenerator: Generator = () => {
  return faker.company.name();
};

const locationGenerator: Generator = (real) => {
  if (/\d/.test(real) && real.includes(',')) {
    return `${faker.location.streetAddress()}, ${faker.location.zipCode()} ${faker.location.city()}`;
  }
  return faker.location.city();
};

const GUID = /^\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?$/i;

function matchCase(fake: string, real: string): string {
  return real === real.toUpperCase() ? fake.toUpperCase() : fake;
}

/** A birth date keeps its real year, so ages and age bands still compute; the day of the year is new. */
const birthDateGenerator: Generator = (real, ctx) => {
  const year = real.match(/\b(1[89]\d\d|20\d\d|2[45]\d\d)\b/)?.[1];
  const fake = dateGenerator(real, ctx);
  return year ? fake.replace(/\b(1[89]\d\d|20\d\d)\b/, year) : fake;
};

const dateGenerator: Generator = (real) => {
  const fake = faker.date.birthdate({ min: 1960, max: 2005, mode: 'year' });
  const sep = real.includes('/') ? '/' : real.includes('-') ? '-' : '.';
  const dayFirst = /^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$/.test(real.trim());
  const d = String(fake.getDate()).padStart(2, '0');
  const m = String(fake.getMonth() + 1).padStart(2, '0');
  const y = String(fake.getFullYear());
  return dayFirst ? `${d}${sep}${m}${sep}${y}` : `${y}${sep}${m}${sep}${d}`;
};

/** An employee or customer code: letters kept, the first two digits kept (they often encode a year), the rest new. */
/** Tries at a fake in the real ID's own cohort before any number of the same length will do. */
const KEEP_COHORT_TRIES = 20;

const employeeIdGenerator: Generator = (real, ctx) => {
  // An internal record id (a GUID) carries no shape worth keeping: a fresh one.
  if (GUID.test(real.trim())) return matchCase(faker.string.uuid(), real);
  const m = real.match(/^([^\d]*)(\d+)(.*)$/);
  if (!m) return formatPreservingGenerator(real, ctx);
  const [, prefix, digits, rest] = m;
  // The first two digits often mark a cohort (a hire year, a site) and are
  // kept while that leaves room; a crowded cohort draws from the whole range.
  const keep = digits.length > 3 && ctx.attempt < KEEP_COHORT_TRIES ? 2 : 0;
  // A range too small or too crowded for one more (badge "7", a placeholder
  // "0") grows a digit at a time rather than run out.
  const grow = ctx.attempt < 200 ? 0 : ctx.attempt < 600 ? 1 : 2;
  return prefix + digits.slice(0, keep) + faker.string.numeric({ length: digits.length - keep + grow, allowLeadingZeros: true }) + rest;
};

/**
 * A value withheld outright rather than faked — an address, a bank account:
 * `[address withheld #3]`. Numbered, so each real value keeps one label.
 */
export function withheld(label: string): Generator {
  return (_real, ctx) => `[${label} withheld #${ctx.count + 1 + ctx.attempt}]`;
}

const formatPreservingGenerator: Generator = (real) => {
  return real.replace(/[A-Z]/g, () => faker.string.alpha({ length: 1, casing: 'upper' }))
             .replace(/[a-z]/g, () => faker.string.alpha({ length: 1, casing: 'lower' }))
             .replace(/[0-9]/g, () => String(faker.number.int({ min: 0, max: 9 })));
};

export const generators: Record<string, Generator> = {
  email: emailGenerator,
  phone: phoneGenerator,
  credit_card: creditCardGenerator,
  ip_address: ipGenerator,
  uuid: uuidGenerator,
  url: urlGenerator,
  tracking_number: trackingGenerator,
  person_name: personNameGenerator,
  first_name: firstNameGenerator,
  last_name: lastNameGenerator,
  organization: organizationGenerator,
  location: locationGenerator,
  date_of_birth: birthDateGenerator,
  date: dateGenerator,
  address: locationGenerator,
  national_id: formatPreservingGenerator,
  employee_id: employeeIdGenerator,
  medical_record: formatPreservingGenerator,
  insurance_id: formatPreservingGenerator,
  passport_number: formatPreservingGenerator,
  license_plate: formatPreservingGenerator,
};

export function getGenerator(type: string): Generator {
  return generators[type] ?? formatPreservingGenerator;
}
