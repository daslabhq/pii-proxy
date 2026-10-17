import { describe, test, expect } from 'bun:test';
import { PrivacyProxy } from '../src/index.js';
import { thai } from '../src/packs/th/index.js';
import { hr } from '../src/packs/hr/index.js';
import { maskRecords } from '../src/files/index.js';

// A whole workforce at once: crowded code cohorts, common nicknames, and
// surnames drawn from the very roots stand-ins are built from.
const NICKNAMES = ['ต้น', 'ทอง', 'แก้ว', 'บอย', 'ปู', 'นก', 'ชัย', 'ดี', 'มณี', 'ส้ม', 'เล็ก', 'น้อย', 'ฝน', 'ฟ้า', 'นุ่น', 'แป้ง', 'มด', 'เก่ง', 'ปาล์ม', 'พลอย'];

function staff(n: number) {
  const source = new PrivacyProxy({ use: [thai(), hr()] });
  return Array.from({ length: n }, (_, i) => {
    const male = i % 2 === 1;
    const first = source.replace(`ชื่อ${i % 600}${male ? 'ช' : 'ญ'}`, 'first_name', { sex: male ? 'male' : 'female' });
    const last = source.replace(`สกุล${i}`, 'last_name');
    const title = male ? 'นาย' : 'นางสาว';
    return {
      EmpCode: `EMP${60 + (i % 3)}${String(Math.floor(i / 3)).padStart(3, '0')}`,
      Title: title, FirstName: first, LastName: last, NickName: NICKNAMES[i % NICKNAMES.length],
      FirstNameEng: `Given${i % 600}`, LastNameEng: `Family${i}`, FullName: `${title}${first} ${last}`,
      DeptNameEng: 'Production', Salary: 15000 + i,
    };
  });
}

describe('a whole workforce', () => {
  test('two thousand people get a stand-in each, quickly, and the same ones again', async () => {
    const rows = staff(2000);
    const proxy = new PrivacyProxy({ use: [thai(), hr()] });
    const started = performance.now();
    const { records } = await maskRecords(rows, proxy);
    expect(performance.now() - started).toBeLessThan(5000);
    expect(new Set(records.map((r) => r.LastName)).size).toBe(2000);
    expect(new Set(records.map((r) => r.EmpCode)).size).toBe(2000);
    expect(new Set(records.map((r) => r.LastNameEng)).size).toBe(2000);
    const again = await maskRecords(rows.slice(0, 50), proxy);
    expect(again.records).toEqual(records.slice(0, 50));
  });
});

describe('a range that runs out', () => {
  test('badge numbers of one digit grow a digit, and nothing is left unmasked', async () => {
    const rows = Array.from({ length: 300 }, (_, i) => ({ EmpCode: `EMP${10000 + i}`, EmpCardNo: String(i % 10), EmpTempCardNo: String(i % 100) }));
    const { records } = await maskRecords(rows, new PrivacyProxy({ use: [hr()] }));
    for (const [i, r] of records.entries()) {
      expect(r.EmpCardNo).not.toBe(rows[i]!.EmpCardNo);
      expect(r.EmpTempCardNo).not.toBe(rows[i]!.EmpTempCardNo);
    }
    expect(new Set(records.map((r) => r.EmpCardNo)).size).toBe(10);
  });

  test('a value no generator can place gets a numbered label rather than failing the batch', () => {
    const proxy = new PrivacyProxy({ generators: { stuck: () => 'same' } });
    const first = proxy.replace('alpha-value', 'stuck');
    const second = proxy.replace('beta-value', 'stuck');
    expect(first).toBe('same');
    expect(second).toBe('[stuck #2]');
    expect(proxy.getMap().getReal('[stuck #2]')).toBe('beta-value');
  });
});
