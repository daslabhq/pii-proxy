import { faker } from '@faker-js/faker';

export type Generator = (real: string) => string;

const emailGenerator: Generator = () => {
  return faker.internet.email().toLowerCase();
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

const ipGenerator: Generator = () => {
  return faker.internet.ipv4();
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

const personNameGenerator: Generator = () => {
  return `${faker.person.firstName()} ${faker.person.lastName()}`;
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

const dateGenerator: Generator = (real) => {
  const fake = faker.date.birthdate({ min: 1960, max: 2005, mode: 'year' });
  const sep = real.includes('/') ? '/' : real.includes('-') ? '-' : '.';
  const dayFirst = /^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$/.test(real.trim());
  const d = String(fake.getDate()).padStart(2, '0');
  const m = String(fake.getMonth() + 1).padStart(2, '0');
  const y = String(fake.getFullYear());
  return dayFirst ? `${d}${sep}${m}${sep}${y}` : `${y}${sep}${m}${sep}${d}`;
};

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
  organization: organizationGenerator,
  location: locationGenerator,
  date_of_birth: dateGenerator,
  date: dateGenerator,
  address: locationGenerator,
  national_id: formatPreservingGenerator,
  medical_record: formatPreservingGenerator,
  insurance_id: formatPreservingGenerator,
  passport_number: formatPreservingGenerator,
  license_plate: formatPreservingGenerator,
};

export function getGenerator(type: string): Generator {
  return generators[type] ?? formatPreservingGenerator;
}
