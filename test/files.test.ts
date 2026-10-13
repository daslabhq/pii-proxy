import { describe, test, expect } from 'bun:test';
import { zipSync, strToU8 } from 'fflate';
import { PrivacyProxy } from '../src/index.js';
import { thai, isThaiNationalId } from '../src/packs/th/index.js';
import { hr } from '../src/packs/hr/index.js';
import { maskFiles, maskTable, maskText, detectColumns, parseCsv, toCsv, readXlsx, excelDate, fileKind } from '../src/files/index.js';

const PAYROLL = [
  ['EmpCode', 'EmpName', 'EmpNameEng', 'EmpTitle', 'EmpFirstName', 'EmpLastName', 'EmpTitleEng', 'EmpFirstNameEng', 'EmpLastNameEng', 'OrgUnitName', '14.ค่าโทรฯ', 'Salary'],
  ['EMP68035', 'นายทดสอบชัย ทองแท้', 'Mr.Thodsobchai  Thongtae', 'นาย', 'ทดสอบชัย', 'ทองแท้', 'Mr.', 'Thodsobchai ', 'Thongtae', 'ฝ่ายคลังสินค้า', '0', '21465'],
  ['EMP66025', 'นางสาวสมมติ ดีงาม', 'Miss.Sommati Deengam', 'นางสาว', 'สมมติ', 'ดีงาม', 'Miss.', 'Sommati', 'Deengam', 'ฝ่ายบัญชี', '500', '30000'],
  ['EMP66026', 'นายทดสอบชัย รักดี', 'Mr.Thodsobchai Rakdee', 'นาย', 'ทดสอบชัย', 'รักดี', 'Mr.', 'Thodsobchai', 'Rakdee', 'ฝ่ายคลังสินค้า', '0', '18000'],
];

const proxy = () => new PrivacyProxy({ use: [thai(), hr()], seed: 7 });

describe('csv and xlsx', () => {
  test('CSV round-trips quotes, commas, newlines and Thai', () => {
    const rows = [['a', 'b,c', 'say "hi"'], ['ไทย', 'line\nbreak', '']];
    expect(parseCsv(toCsv(rows))).toEqual(rows);
  });

  test('a semicolon CSV is read by its own delimiter', () => {
    expect(parseCsv('name;salary\nSom;100\n')).toEqual([['name', 'salary'], ['Som', '100']]);
  });

  test('Excel serial dates become ISO dates', () => {
    expect(excelDate(46295)).toBe('2026-09-30');
    expect(excelDate(46295.5)).toBe('2026-09-30 12:00:00');
  });

  test('an .xlsx is read: shared and inline strings, numbers at shortest form, dates', () => {
    const xlsx = zipSync({
      'xl/workbook.xml': strToU8('<workbook><sheets><sheet name="Pay" sheetId="1" r:id="rId1"/></sheets></workbook>'),
      'xl/_rels/workbook.xml.rels': strToU8('<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'),
      'xl/sharedStrings.xml': strToU8('<sst><si><t>EmpName</t></si><si><t>Pay&amp;Date</t></si><si><r><t>นายทดสอบ</t></r><r><t>ชัย</t></r></si></sst>'),
      'xl/styles.xml': strToU8('<styleSheet><cellXfs><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>'),
      'xl/worksheets/sheet1.xml': strToU8(
        '<worksheet><sheetData>' +
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="inlineStr"><is><t>Net</t></is></c></row>' +
        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" s="1"><v>46295</v></c><c r="C2"><v>10231.719999999999</v></c></row>' +
        '</sheetData></worksheet>'),
    });
    const [sheet] = readXlsx(xlsx);
    expect(sheet!.name).toBe('Pay');
    expect(sheet!.rows).toEqual([['EmpName', 'Pay&Date', 'Net'], ['นายทดสอบชัย', '2026-09-30', '10231.72']]);
  });
});

describe('columns', () => {
  test('payroll headers: names, titles and codes; an allowance named after the phone is not a phone', async () => {
    const types = Object.fromEntries((await detectColumns(PAYROLL, proxy())).columns.map((c) => [c.header, c.type]));
    expect(types).toEqual({
      EmpCode: 'employee_id', EmpName: 'person_name', EmpNameEng: 'person_name', EmpTitle: 'title',
      EmpFirstName: 'first_name', EmpLastName: 'last_name', EmpTitleEng: 'title',
      EmpFirstNameEng: 'first_name', EmpLastNameEng: 'last_name',
    });
  });

  test('people are counted by code', async () => {
    expect((await detectColumns(PAYROLL, proxy())).people).toBe(3);
  });

  test('department, position and company names are not people', async () => {
    const rows = [['OrgUnitName', 'PositionNameEng', 'CostCenterName', 'OrgNameEng', 'Social'], ['Sales', 'Clerk', 'CC1', 'Acme', '750']];
    expect((await detectColumns(rows, proxy())).columns).toEqual([]);
  });

  test('Thai headers come from the Thai pack', async () => {
    const rows = [['ชื่อ-นามสกุล', 'นามสกุล', 'รหัสพนักงาน', 'เบอร์โทรศัพท์', '14.ค่าโทรฯ'], ['นายทดสอบชัย ทองแท้', 'ทองแท้', 'EMP1', '081-234-5678', '500']];
    const types = (await detectColumns(rows, proxy())).columns.map((c) => c.type);
    expect(types).toEqual(['person_name', 'last_name', 'employee_id', 'phone']);
    expect((await detectColumns(rows, new PrivacyProxy())).columns).toEqual([]);
  });

  test('an unlabeled column of emails is found by its values', async () => {
    const rows = [['x'], ['a@b.co'], ['c@d.co'], ['e@f.co']];
    expect((await detectColumns(rows, proxy())).columns).toEqual([{ index: 0, header: 'x', type: 'email' }]);
  });
});

describe('tables', () => {
  test('names and codes are masked; everything else is untouched', async () => {
    const { rows } = await maskTable(PAYROLL, proxy());
    expect(rows[0]).toEqual(PAYROLL[0]!);
    for (let r = 1; r < rows.length; r++) {
      for (const c of [3, 6, 9, 10, 11]) expect(rows[r]![c]).toBe(PAYROLL[r]![c]!);
      for (const c of [0, 1, 2, 4, 5, 7, 8]) expect(rows[r]![c]).not.toBe(PAYROLL[r]![c]!);
    }
  });

  test('fakes follow the real value: Thai for Thai, English for English, the title picks the sex, the code keeps its shape', async () => {
    const { rows } = await maskTable(PAYROLL, proxy());
    expect(rows[1]![4]).toMatch(/[฀-๿]/);
    expect(rows[1]![7]).toMatch(/^[A-Za-z' -]+\s$/);
    expect(rows[1]![0]).toMatch(/^EMP68\d{3}$/);
  });

  test('a full name is rebuilt from its own masked parts, in the same layout', async () => {
    const { rows } = await maskTable(PAYROLL, proxy());
    const r = rows[1]!;
    expect(r[1]).toBe(`นาย${r[4]} ${r[5]}`);
    expect(r[2]).toBe(`Mr.${r[7]} ${r[8]}`);
  });

  test('two people sharing a first name share its fake; their surnames stay apart', async () => {
    const { rows } = await maskTable(PAYROLL, proxy());
    expect(rows[1]![4]).toBe(rows[3]![4]!);
    expect(rows[1]![5]).not.toBe(rows[3]![5]!);
  });
});

describe('files', () => {
  test('a batch: one CSV per file, no real value anywhere, the same person the same fake in both files', async () => {
    const p = proxy();
    const bytes = (rows: string[][]) => new TextEncoder().encode(toCsv(rows));
    const { results, skipped } = await maskFiles([
      { name: 'pay.csv', bytes: bytes(PAYROLL) },
      { name: 'bonus.csv', bytes: bytes([PAYROLL[0]!, PAYROLL[2]!]) },
      { name: 'scan.pdf', bytes: new Uint8Array([37, 80, 68, 70]) },
    ], p);
    expect(results.map((r) => r.files[0]!.name)).toEqual(['pay (masked).csv', 'bonus (masked).csv']);
    expect(skipped).toEqual([{ source: 'scan.pdf', reason: 'only .xlsx, .csv and text files can be masked' }]);
    const [pay, bonus] = results.map((r) => parseCsv(new TextDecoder().decode(r.files[0]!.bytes)));
    expect(bonus![1]).toEqual(pay![2]!);
    const all = results.map((r) => new TextDecoder().decode(r.files[0]!.bytes)).join('\n');
    for (const [real] of p.getMap().entries()) expect(all.includes(real)).toBe(false);
    expect(results[0]!.columns.map((c) => c.header)).toContain('EmpCode');
  });

  test('a map saved and loaded gives next month the same fakes', async () => {
    const first = proxy();
    const sep = await maskTable(PAYROLL, first);
    const next = new PrivacyProxy({ use: [thai(), hr()], seed: 99 });
    next.loadMap(first.getMap().serialize());
    const oct = await maskTable(PAYROLL, next);
    expect(oct.rows).toEqual(sep.rows);
  });

  test('text: known whole names, emails and national IDs are masked', async () => {
    const p = proxy();
    await maskTable(PAYROLL, p);
    const id = '1-1037-01234-56-3';
    expect(isThaiNationalId(id)).toBe(true);
    const { text } = await maskText(`นายทดสอบชัย ทองแท้ (${id}) mailed somchai@acme.example`, p);
    expect(text).not.toContain('ทองแท้');
    expect(text).not.toContain(id);
    expect(text).not.toContain('somchai@acme.example');
  });

  test('PDFs and images are not readable yet', () => {
    expect(fileKind('scan.pdf', 'application/pdf')).toBeNull();
    expect(fileKind('photo.jpg', 'image/jpeg')).toBeNull();
    expect(fileKind('pay.xlsx')).toBe('workbook');
  });
});
