import { describe, test, expect } from 'bun:test';
import { PrivacyProxy } from '../src/index.js';
import { thai, isThaiNationalId } from '../src/packs/th/index.js';

describe('thai pack', () => {
  test('titles pick the fake\'s sex — Thai from the pack, English from the core', () => {
    const p = new PrivacyProxy({ use: [thai()] });
    expect(p.sexOfTitle('นาย')).toBe('male');
    expect(p.sexOfTitle('นางสาว')).toBe('female');
    expect(p.sexOfTitle('Mr.')).toBe('male');
    expect(p.sexOfTitle('Miss')).toBe('female');
    expect(p.sexOfTitle('ดร.')).toBeUndefined();
  });

  test('a Thai national ID is found in text and faked with a valid check digit, same grouping', async () => {
    const p = new PrivacyProxy({ use: [thai()], seed: 3 });
    const { text, detections } = await p.mask('บัตร 1-1037-01234-56-3 ครับ');
    expect(detections.map((d) => d.type)).toEqual(['national_id']);
    const fake = detections[0]!.replacement;
    expect(fake).toMatch(/^\d-\d{4}-\d{5}-\d{2}-\d$/);
    expect(isThaiNationalId(fake)).toBe(true);
    expect(text).toBe(`บัตร ${fake} ครับ`);
  });

  test('one real value, one fake — stable, and never containing a real value', () => {
    const p = new PrivacyProxy({ use: [thai()], seed: 7 });
    p.reserve(['ทดสอบชัย', 'Thodsobchai', 'Thongtae']);
    const a = p.replace('ทดสอบชัย', 'first_name', { sex: 'male' });
    expect(p.replace('ทดสอบชัย', 'first_name')).toBe(a);
    expect(p.getMap().getReal(a)).toBe('ทดสอบชัย');
    expect(a).not.toContain('ทดสอบชัย');
  });

  test('a large roster still gets distinct, name-shaped fakes in both scripts', () => {
    const p = new PrivacyProxy({ use: [thai()], seed: 7 });
    const thaiFirsts = Array.from({ length: 300 }, (_, i) => p.replace(`ชื่อ${i}`, 'first_name', { sex: i % 2 ? 'male' : 'female' }));
    const englishLasts = Array.from({ length: 400 }, (_, i) => p.replace(`Family${i}`, 'last_name'));
    expect(new Set(thaiFirsts).size).toBe(300);
    expect(new Set(englishLasts).size).toBe(400);
    expect(thaiFirsts.every((n) => /^[฀-๿]+$/.test(n))).toBe(true);
  });

  test('a phone keeps its format and network prefix', () => {
    const p = new PrivacyProxy({ use: [thai()], seed: 1 });
    expect(p.replace('081-234-5678', 'phone')).toMatch(/^08\d-\d{3}-\d{4}$/);
  });
});
