import { describe, test, expect } from 'bun:test';
import { PrivacyProxy } from '../src/index.js';
import { thai } from '../src/packs/th/index.js';
import { hr } from '../src/packs/hr/index.js';
import { maskRecords } from '../src/files/index.js';

// Shaped like an HR system's employee view: identity, org, pay, and a manager reference.
const staff = {
  EmpID: '3F2504E0-4F89-11D3-9A0C-0305E82C3301', EmpCode: 'EMP68035',
  Title: 'นาย', FirstName: 'ทดสอบชัย', MiddleName: null, LastName: 'ทองแท้', NickName: 'ต้นกล้า',
  TitleEng: 'Mr.', FirstNameEng: 'Thodsobchai', LastNameEng: 'Thongtae', FullName: 'นายทดสอบชัย ทองแท้',
  IdentityCard: '1-1037-01234-56-3', BirthDate: new Date('1985-03-12T00:00:00Z'), WorkPermit: null,
  ReportToEmpCode: 'EMP66025', ReportToEmpFullName: 'นางสาวสมมติ ดีงาม',
  DeptNameEng: 'Warehouse', PositionCode: '05-114', Salary: 21465, StartDate: new Date('2019-06-01T00:00:00Z'),
};
const manager = {
  EmpID: '7A2B11C0-0000-4000-8000-00000000AB12', EmpCode: 'EMP66025',
  Title: 'นางสาว', FirstName: 'สมมติ', MiddleName: null, LastName: 'ดีงาม', NickName: 'ฝนพรำ',
  TitleEng: 'Miss', FirstNameEng: 'Sommati', LastNameEng: 'Deengam', FullName: 'นางสาวสมมติ ดีงาม',
  IdentityCard: '3-1005-00876-54-1', BirthDate: new Date('1979-11-02T00:00:00Z'), WorkPermit: null,
  ReportToEmpCode: null, ReportToEmpFullName: '',
  DeptNameEng: 'Warehouse', PositionCode: '03-020', Salary: 48000, StartDate: new Date('2012-01-09T00:00:00Z'),
};
const proxy = () => new PrivacyProxy({ use: [thai(), hr()], seed: 11 });

describe('records', () => {
  test('every personal column is masked; org, pay and dates are untouched and keep their types', async () => {
    const { records } = await maskRecords([staff, manager], proxy());
    const [s] = records as any[];
    for (const k of ['EmpID', 'EmpCode', 'FirstName', 'LastName', 'NickName', 'FirstNameEng', 'LastNameEng', 'FullName', 'IdentityCard', 'ReportToEmpCode', 'ReportToEmpFullName']) {
      expect(s[k]).not.toBe((staff as any)[k]);
    }
    expect(s.DeptNameEng).toBe('Warehouse');
    expect(s.PositionCode).toBe('05-114');
    expect(s.Salary).toBe(21465);
    expect(s.StartDate).toBe(staff.StartDate);
    expect(s.MiddleName).toBeNull();
    expect(s.Title).toBe('นาย');
  });

  test('a GUID gets a GUID; a code keeps its shape; a birth date keeps its year', async () => {
    const [s] = (await maskRecords([staff], proxy())).records as any[];
    expect(s.EmpID).toMatch(/^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$/);
    expect(s.EmpCode).toMatch(/^EMP68\d{3}$/);
    expect(String(s.BirthDate)).toMatch(/^1985-\d{2}-\d{2}/);
  });

  test('a manager gets one fake, on their own row and on their report\'s', async () => {
    const [s, m] = (await maskRecords([staff, manager], proxy())).records as any[];
    expect(s.ReportToEmpCode).toBe(m.EmpCode);
    expect(s.ReportToEmpFullName).toBe(m.FullName);
    expect(m.FullName).toBe(`นางสาว${m.FirstName} ${m.LastName}`);
  });

  test('a name under a header nobody anticipated is caught when the map knows it', async () => {
    const p = proxy();
    await maskRecords([manager], p);
    const [row] = (await maskRecords([{ ApproveBy: 'นางสาวสมมติ ดีงาม', Hours: 3 }], p)).records as any[];
    expect(row.ApproveBy).not.toContain('สมมติ');
    expect(row.Hours).toBe(3);
  });

  test('an aggregate passes through exactly as it came', async () => {
    const rows = [{ OrgCode: 'A1', dept_name: 'Warehouse', people: 9, gross: 201345.5 }];
    expect((await maskRecords(rows, proxy())).records).toEqual(rows);
  });
});
