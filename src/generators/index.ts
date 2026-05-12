/**
 * Fake value generators — produce plausible replacements for each entity type.
 *
 * Uses @faker-js/faker for realistic output. Format-preserving where possible
 * so LLMs don't get confused by mismatched formats.
 */

import { faker } from '@faker-js/faker';
import type { EntityType } from '../detectors/index.js';

export type Generator = (real: string) => string;

const emailGenerator: Generator = () => {
  return faker.internet.email().toLowerCase();
};

const phoneGenerator: Generator = (real) => {
  // Preserve format: if real has +XX prefix, generate with prefix
  const hasPlus = real.startsWith('+');
  const digits = (real.match(/\d/g) || []).length;
  const phone = faker.phone.number();
  if (hasPlus && !phone.startsWith('+')) return '+' + phone;
  return phone;
};

const creditCardGenerator: Generator = (real) => {
  // Preserve card type (first digit) and length
  const clean = real.replace(/\D/g, '');
  const separator = real.includes('-') ? '-' : real.includes(' ') ? ' ' : '';
  const fake = faker.finance.creditCardNumber();
  if (!separator) return fake.replace(/\D/g, '');
  // Reformat to match original grouping
  const groups = real.split(/[-\s]/);
  const fakeClean = fake.replace(/\D/g, '');
  let pos = 0;
  return groups.map(g => {
    const chunk = fakeClean.slice(pos, pos + g.length);
    pos += g.length;
    return chunk;
  }).join(separator);
};

const ipGenerator: Generator = () => {
  return faker.internet.ipv4();
};

const uuidGenerator: Generator = () => {
  return faker.string.uuid();
};

const urlGenerator: Generator = (real) => {
  // Keep the domain structure, randomize path/query
  try {
    const url = new URL(real);
    return `https://example.com/${faker.string.alphanumeric(8)}?token=${faker.string.alphanumeric(16)}`;
  } catch {
    return `https://example.com/${faker.string.alphanumeric(20)}`;
  }
};

const trackingGenerator: Generator = (real) => {
  // Format-preserving: match the length and character pattern
  const result = real.replace(/[A-Z]/g, () => faker.string.alpha({ length: 1, casing: 'upper' }))
                     .replace(/[0-9]/g, () => String(faker.number.int({ min: 0, max: 9 })));
  return result;
};

const defaultGenerator: Generator = (real) => {
  // Fallback: replace with same-length alphanumeric
  return faker.string.alphanumeric(real.length);
};

/** Map of entity type → generator function. */
export const generators: Record<EntityType, Generator> = {
  email: emailGenerator,
  phone: phoneGenerator,
  credit_card: creditCardGenerator,
  ip_address: ipGenerator,
  uuid: uuidGenerator,
  url: urlGenerator,
  tracking_number: trackingGenerator,
  date: defaultGenerator,
  address: defaultGenerator,
};
