import { describe, test, expect } from 'bun:test';
import { BijectiveMap } from '../src/map.js';
import { knownValues, replaceKnown } from '../src/index.js';

function map(entries: Array<[string, string, string?]>): BijectiveMap {
  const m = new BijectiveMap();
  for (const [real, fake, type] of entries) m.set(real, fake, type);
  return m;
}

describe('replaceKnown', () => {
  const people = map([
    ['นายทดสอบชัย ทองแท้', 'นายพีระพงษ์ แสงทอง', 'person_name'],
    ['Thodsobchai Thongtae', 'Peerapong Saengthong', 'person_name'],
    ['ทดสอบชัย', 'พีระพงษ์', 'first_name'],
    ['EMP68035', 'EMP68417', 'employee_id'],
    ['1-1037-01234-56-3', '1-2040-55555-12-7', 'national_id'],
  ]);
  const scrub = replaceKnown(people);

  test('full names and codes anywhere in free text become their fakes', () => {
    expect(scrub('OT for นายทดสอบชัย ทองแท้ (EMP68035), id 1-1037-01234-56-3; cc Thodsobchai Thongtae'))
      .toBe('OT for นายพีระพงษ์ แสงทอง (EMP68417), id 1-2040-55555-12-7; cc Peerapong Saengthong');
  });

  test('a lone first name is left alone: it is a word as often as a person', () => {
    expect(scrub('ทดสอบชัย said hi')).toBe('ทดสอบชัย said hi');
  });

  test('a code inside a longer code is not a match', () => {
    expect(scrub('EMP680351 and XEMP68035')).toBe('EMP680351 and XEMP68035');
  });

  test('a short run of digits is an amount as often as an ID, and stays', () => {
    const m = map([
      ['158858', '204417', 'employee_id'],
      ['1103701234563', '1204055555127', 'national_id'],
    ]);
    expect(replaceKnown(m)('total 158858.00, id 1103701234563')).toBe('total 158858.00, id 1204055555127');
  });

  test('text the map has never met comes back as the same string', () => {
    const text = 'Warehouse, 21,465 THB';
    expect(scrub(text)).toBe(text);
  });

  test('a real value that is also a stand-in stays, so no stand-in is renamed', () => {
    const m = map([
      ['EMP10001', 'EMP20002', 'employee_id'],
      ['EMP20002', 'EMP30003', 'employee_id'],
    ]);
    expect(replaceKnown(m)('EMP10001 EMP20002')).toBe('EMP20002 EMP20002');
  });

  test('the longest value wins where two start at the same place', () => {
    const m = map([
      ['Somsak Rakdee', 'Anan Boonmee', 'person_name'],
      ['Somsak Rakdee-Wong', 'Anan Boonmee-Sri', 'person_name'],
    ]);
    expect(replaceKnown(m)('Somsak Rakdee-Wong, Somsak Rakdee')).toBe('Anan Boonmee-Sri, Anan Boonmee');
  });

  test('values the map learns later are found too', () => {
    const m = map([['EMP10001', 'EMP20002', 'employee_id']]);
    const s = replaceKnown(m);
    expect(s('EMP40004')).toBe('EMP40004');
    m.set('EMP40004', 'EMP50005', 'employee_id');
    expect(s('EMP40004')).toBe('EMP50005');
  });

  test('thousands of people over a long text stay fast', () => {
    const m = new BijectiveMap();
    for (let i = 0; i < 5000; i++) {
      m.set(`EMP${10000 + i}`, `EMP${50000 + i}`, 'employee_id');
      m.set(`Person${i} Surname${i}`, `Fake${i} Other${i}`, 'person_name');
    }
    const text = Array.from({ length: 2000 }, (_, i) => `row ${i}: Person${i} Surname${i} EMP${10000 + i} Warehouse`).join('\n');
    const started = performance.now();
    const out = replaceKnown(m)(text);
    expect(performance.now() - started).toBeLessThan(500);
    expect(out).toContain('row 7: Fake7 Other7 EMP50007 Warehouse');
  });
});

describe('knownValues', () => {
  test('reports each find once, without overlaps, with its type', () => {
    const m = map([
      ['นายทดสอบชัย ทองแท้', 'นายพีระพงษ์ แสงทอง', 'person_name'],
      ['ทดสอบชัย ทองแท้', 'พีระพงษ์ แสงทอง', 'person_name'],
    ]);
    const found = knownValues(m).detect('ถึง นายทดสอบชัย ทองแท้') as Array<{ type: string; value: string; start: number }>;
    expect(found).toEqual([{ type: 'person_name', value: 'นายทดสอบชัย ทองแท้', start: 4, end: 22 } as any]);
  });
});
