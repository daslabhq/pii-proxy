#!/usr/bin/env bun
/**
 * Build the Thai evaluation set: data/thai.jsonl, same record format as
 * build_data.py ({id, text, spans: [{start, end, label}], split, domain}).
 *
 * No public Thai PII benchmark was found, so this one is generated: Thai
 * names, national IDs (valid check digit) and phone numbers come from the
 * Thai pack's own generators, English names from the core generators, all
 * seeded. They fill templates written the way these texts are written —
 * LINE group messages, HR emails, payroll remarks, Thai mixed with English.
 * No real person is in it.
 *
 * Span conventions: a name excludes its title (นาย, นาง, นางสาว, คุณ, พี่,
 * น้อง, Mr) — titles are written glued to the name in Thai, as they are here.
 * A name may also be glued to the word after it (ขอบคุณคุณสมชายมากครับ),
 * since Thai puts no space between words.
 *
 *   bun experiments/007-detector-benchmark/build_thai.ts
 */
import { faker } from '@faker-js/faker';
import { writeFileSync } from 'fs';
import { join } from 'path';
import { thai } from '../../src/packs/th/index.js';
import type { GeneratorContext } from '../../src/generators/index.js';

const SEED = 42;
const FILLS_PER_TEMPLATE = 9;

const TEMPLATES: Array<[genre: string, template: string]> = [
  // LINE group messages
  ['line', 'คุณ{FIRST}คะ รบกวนโทรกลับเบอร์ {PHONE} ด้วยนะคะ'],
  ['line', 'พรุ่งนี้{TITLE}{NAME} ขอลาป่วยครับ ถ้ามีงานด่วนโทร {PHONE} ได้เลย'],
  ['line', 'ส่งสำเนาบัตรประชาชนของ{TITLE}{NAME} เลขที่ {ID} มาในกลุ่มแล้วนะครับ'],
  ['line', '@{FIRST} ช่วยเช็คสลิปเงินเดือนเดือนนี้ให้หน่อยค่ะ รหัสพนักงาน {EMP}'],
  ['line', 'ใครมีเบอร์คุณ{FIRST}บ้างคะ เบอร์เดิม {PHONE} โทรไม่ติดเลย'],
  ['line', 'แจ้งเปลี่ยนเบอร์ติดต่อของ{TITLE}{NAME} เป็น {PHONE} นะครับ'],
  ['line', 'พี่{FIRST} ฝากบอก{TITLE2}{NAME2} ว่าเอกสารเซ็นเรียบร้อยแล้วค่ะ'],
  ['line', 'น้อง{FIRST} เริ่มงานวันจันทร์นี้ค่ะ รหัสพนักงาน {EMP} อีเมล {EMAIL}'],
  ['line', 'ขอบคุณคุณ{FIRST}มากครับ ได้รับเอกสารแล้ว'],
  ['line', 'เลขบัตร {ID} ของ{TITLE}{NAME} ไม่ตรงกับในระบบ ช่วยตรวจสอบหน่อยครับ'],
  ['line', 'กลุ่ม HR: {TITLE}{NAME} แจ้งเปลี่ยนนามสกุลหลังแต่งงาน ส่งเลขบัตร {ID} มาแล้วค่ะ'],
  ['line', 'ฝากเบอร์ {PHONE} ของคุณ{FIRST} แผนกจัดซื้อไว้ด้วยนะ'],
  ['line', 'ตอนนี้{TITLE}{NAME}อยู่โรงพยาบาล ญาติแจ้งมาที่เบอร์ {PHONE}'],
  // HR emails
  ['email', 'เรียน คุณ{NAME}\n\nฝ่ายทรัพยากรบุคคลขอแจ้งว่าท่านผ่านการทดลองงานแล้ว มีผลตั้งแต่วันที่ 1 พฤศจิกายน 2569 หากมีข้อสงสัยกรุณาติดต่อ{TITLE2}{NAME2} โทร {PHONE} หรืออีเมล {EMAIL}\n\nขอแสดงความนับถือ\nฝ่ายทรัพยากรบุคคล'],
  ['email', 'เรียน ผู้จัดการฝ่ายบัญชี\n\nขออนุมัติปรับเงินเดือนของ{TITLE}{NAME} รหัสพนักงาน {EMP} ตามผลการประเมินประจำปี รายละเอียดตามเอกสารแนบ\n\nด้วยความเคารพ\n{NAME2}'],
  ['email', 'เรื่อง ขอหนังสือรับรองเงินเดือน\n\nข้าพเจ้า{TITLE}{NAME} เลขประจำตัวประชาชน {ID} พนักงานแผนกขาย ขอหนังสือรับรองเงินเดือนเพื่อใช้ยื่นกู้ธนาคาร ติดต่อกลับได้ที่ {PHONE}'],
  ['email', 'แจ้งพนักงานใหม่: {TITLE}{NAME} ตำแหน่งเจ้าหน้าที่คลังสินค้า เริ่มงาน 3 พ.ย. 2569 รหัส {EMP} อีเมลบริษัท {EMAIL}'],
  ['email', 'เรียน คุณ{FIRST}\n\nตามที่ท่านแจ้งลาออก มีผลวันที่ 30 พฤศจิกายน 2569 ฝ่ายบุคคลจะโอนเงินชดเชยเข้าบัญชีเดิม กรุณายืนยันเลขบัตรประชาชน {ID} ภายในสัปดาห์นี้'],
  ['email', 'สวัสดีครับ คุณ{NAME}\n\nผมชื่อ{NAME2} จากฝ่ายบุคคล ขอนัดสัมภาษณ์วันพุธที่ 21 ตุลาคม เวลา 10.00 น. หากไม่สะดวกโทรแจ้งที่ {PHONE} ได้ครับ'],
  ['email', 'ผู้ติดต่อกรณีฉุกเฉินของ{TITLE}{NAME}: {TITLE2}{NAME2} (คู่สมรส) โทร {PHONE}'],
  // Payroll remarks
  ['payroll', '{EMP} {TITLE}{NAME} หักประกันสังคม 750 บาท'],
  ['payroll', 'โอนเงินเดือนเข้าบัญชี{TITLE}{NAME} แล้ว ({EMP})'],
  ['payroll', '{TITLE}{NAME} ({ID}) ขาดงาน 2 วัน หักตามระเบียบ'],
  ['payroll', 'ค่าล่วงเวลา {EMP} {FIRST} 12 ชม.'],
  ['payroll', 'ยกเลิกรายการซ้ำของ{TITLE}{NAME} เลขบัตร {ID}'],
  ['payroll', 'ปรับฐานเงินเดือน {NAME} ตามคำสั่งที่ 45/2569'],
  ['payroll', '{NAME} โทร {PHONE} ขอเลื่อนวันโอน'],
  ['payroll', 'เบิกค่าเดินทาง {TITLE}{NAME} รหัส {EMP} รวม 1,250 บาท'],
  // Thai mixed with English
  ['mixed', 'Hi {EN_NAME}, please update the record for {TITLE}{NAME} (ID {ID}), new phone {PHONE}.'],
  ['mixed', 'Payroll note: คุณ{FIRST} ({EMP}) ย้ายไปสาขาบางนา from November, contact {EMAIL}'],
  ['mixed', 'FYI {EN_NAME} — {TITLE}{NAME} ส่ง resignation letter แล้วครับ last day 30 Nov'],
  ['mixed', 'Please call {TITLE}{NAME} at {PHONE} to confirm the onboarding date ค่ะ'],
  ['mixed', 'Dear {EN_NAME},\n\nAttached is the ID copy of {NAME} ({ID}) for the visa application. ถ้าต้องการเอกสารเพิ่มเติม ติดต่อ {EMAIL} ได้เลยครับ'],
  ['mixed', 'ส่ง payslip ให้ {NAME} ทางอีเมล {EMAIL} แล้ว cc {EN_NAME}'],
  ['mixed', 'Meeting with คุณ{FIRST} และ {EN_NAME} พรุ่งนี้ 2pm, เบอร์ติดต่อ {PHONE}'],
  ['mixed', 'Employee {EMP} {NAME} — tax ID {ID} — ส่งเอกสารลดหย่อนครบแล้ว'],
];

const SLOT_LABEL: Record<string, string | null> = {
  NAME: 'NAME', NAME2: 'NAME', FIRST: 'NAME', EN_NAME: 'NAME',
  ID: 'NATIONAL_ID', PHONE: 'PHONE', EMAIL: 'EMAIL', EMP: 'EMPLOYEE_ID',
  TITLE: null, TITLE2: null,
};

const ROMANISED = ['somchai', 'suda', 'nattapong', 'kanokwan', 'wichai', 'pornthip', 'thanakorn',
  'siriporn', 'anucha', 'jiraporn', 'kittipong', 'nuttida', 'prasert', 'wanpen', 'surachai', 'ratana'];
const DOMAINS = ['gmail.com', 'hotmail.com', 'outlook.co.th', 'siamtrading.co.th', 'example.co.th'];
const PHONE_SHAPES = ['081-234-5678', '0812345678', '08-1234-5678', '+66 81 234 5678', '02-123-4567',
  '091-234-5678', '062 345 6789'];
const ID_SHAPES = ['1-2345-67890-12-1', '1 2345 67890 12 1', '1234567890121'];

function main(): void {
  faker.seed(SEED);
  const rng = mulberry32(SEED);
  const records = [];
  for (const [t, [genre, template]] of TEMPLATES.entries()) {
    for (let k = 0; k < FILLS_PER_TEMPLATE; k++) {
      const { text, spans } = fill(template, slotValues(rng));
      records.push({ id: `thai-${t}-${k}`, text, spans, split: 'test', domain: genre });
    }
  }
  const out = join(import.meta.dir, 'data', 'thai.jsonl');
  writeFileSync(out, records.map((r) => JSON.stringify(r)).join('\n') + '\n');
  const counts: Record<string, number> = {};
  for (const r of records) for (const s of r.spans) counts[s.label] = (counts[s.label] ?? 0) + 1;
  console.log(`thai: ${records.length} records -> ${out}`, counts);
}

function slotValues(rng: () => number): Record<string, string> {
  const pack = thai().generators!;
  const ctx = (sex: 'male' | 'female'): GeneratorContext =>
    ({ attempt: Math.floor(rng() * 20), rng, hints: { sex }, count: 0 });
  const person = (sex: 'male' | 'female') => ({
    first: pack.first_name!('สมชาย', ctx(sex)),
    last: pack.last_name!('ใจดี', ctx(sex)),
    title: sex === 'male' ? 'นาย' : pick(['นาง', 'นางสาว'], rng),
  });
  const a = person(rng() < 0.5 ? 'male' : 'female');
  const b = person(rng() < 0.5 ? 'male' : 'female');
  return {
    NAME: `${a.first} ${a.last}`, FIRST: a.first, TITLE: a.title,
    NAME2: `${b.first} ${b.last}`, TITLE2: b.title,
    EN_NAME: `${faker.person.firstName()} ${faker.person.lastName()}`,
    ID: pack.national_id!(pick(ID_SHAPES, rng), ctx('male')),
    PHONE: pack.phone!(pick(PHONE_SHAPES, rng), ctx('male')),
    EMAIL: `${pick(ROMANISED, rng)}${pick(['.', '_', ''], rng)}${Math.floor(rng() * 900 + 10)}@${pick(DOMAINS, rng)}`,
    EMP: rng() < 0.6 ? `EMP${Math.floor(rng() * 90000 + 10000)}` : `E-${Math.floor(rng() * 9000 + 1000)}`,
  };
}

function fill(template: string, values: Record<string, string>) {
  let text = '';
  const spans: Array<{ start: number; end: number; label: string }> = [];
  for (const part of template.split(/(\{[A-Z_0-9]+\})/)) {
    const slot = part.match(/^\{([A-Z_0-9]+)\}$/)?.[1];
    if (!slot) { text += part; continue; }
    const value = values[slot]!;
    const label = SLOT_LABEL[slot];
    if (label) spans.push({ start: text.length, end: text.length + value.length, label });
    text += value;
  }
  return { text, spans };
}

function pick<T>(list: readonly T[], rng: () => number): T {
  return list[Math.floor(rng() * list.length)]!;
}

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

main();
