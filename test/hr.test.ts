import { describe, test, expect } from 'bun:test';
import { PrivacyProxy } from '../src/index.js';
import { hr } from '../src/packs/hr/index.js';
import { thai } from '../src/packs/th/index.js';
import { detectColumns, maskTable } from '../src/files/index.js';

const ROSTER = [
  ['EmpCode', 'FirstName', 'LastName', 'Address', 'Bank Account', 'Prepared By', 'Department', 'Salary'],
  ['EMP68035', 'Dana', 'Whitlow', '12 Elm Road', '123-4-56789-0', 'Pat Okafor', 'Finance', '4100'],
  ['EMP68036', 'Remy', 'Castell', '9 Oak Lane', '987-6-54321-0', 'Pat Okafor', 'Finance', '3900'],
];

describe('hr pack', () => {
  test('employee codes and sign-off columns are personal only with the pack', async () => {
    const types = async (p: PrivacyProxy) => Object.fromEntries((await detectColumns(ROSTER, p)).columns.map((c) => [c.header, c.type]));
    expect(await types(new PrivacyProxy({ use: [hr()] }))).toEqual({
      EmpCode: 'employee_id', FirstName: 'first_name', LastName: 'last_name',
      Address: 'address', 'Bank Account': 'bank_account', 'Prepared By': 'person_name',
    });
    expect(await types(new PrivacyProxy())).toEqual({
      FirstName: 'first_name', LastName: 'last_name', Address: 'address', 'Bank Account': 'bank_account',
    });
  });

  test('addresses and bank accounts are withheld, one label per value; codes keep their shape', async () => {
    const { rows } = await maskTable(ROSTER, new PrivacyProxy({ use: [hr()], seed: 7 }));
    expect(rows[1]![3]).toMatch(/^\[address withheld #\d+\]$/);
    expect(rows[1]![4]).toMatch(/^\[bank account withheld #\d+\]$/);
    expect(rows[2]![4]).not.toBe(rows[1]![4]);
    expect(rows[1]![0]).toMatch(/^EMP68\d{3}$/);
    expect(rows[1]![5]).toBe(rows[2]![5]!);
    expect(rows[1]![5]).not.toBe('Pat Okafor');
    expect(rows[1]!.slice(6)).toEqual(['Finance', '4100']);
  });

  test('combines with a locale: Thai words for the same columns', async () => {
    const rows = [['รหัสพนักงาน', 'เลขที่บัญชี'], ['EMP1', '123-4-56789-0']];
    const { rows: out } = await maskTable(rows, new PrivacyProxy({ use: [thai(), hr()], seed: 1 }));
    expect(out[1]![1]).toMatch(/^\[bank account withheld #\d+\]$/);
  });
});
