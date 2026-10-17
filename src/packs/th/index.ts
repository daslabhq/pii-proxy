/**
 * Thai pack — Thai names for Thai names, Thai national IDs found and faked
 * with a valid check digit, Thai phone numbers kept on their network prefix.
 *
 *   import { PrivacyProxy } from 'pii-proxy';
 *   import { thai } from 'pii-proxy/packs/th';
 *   const proxy = new PrivacyProxy({ use: [thai()] });
 *
 * Names follow the script of the real value: a Thai name gets a Thai fake,
 * a name in Latin script gets an English one, so a payroll that carries both
 * stays consistent. Pass `hints.sex` (proxy.sexOfTitle reads it off นาย /
 * นาง / นางสาว as well as Mr / Mrs / Miss) and the fake follows it. The lists
 * run out on a large roster; later attempts compose new names from parts, as
 * Thai names are composed, so a fake never falls back to a code.
 *
 * For tables (`pii-proxy/files`) the pack also knows Thai column headers —
 * ชื่อ, นามสกุล, รหัสพนักงาน, เลขบัตรประชาชน — and that a header starting
 * with ค่า names a fee or an allowance (ค่าโทร is a phone allowance, an
 * amount), never a contact detail.
 */
import { generators as base, type Generator } from '../../generators/index.js';
import type { Detector } from '../../detectors/index.js';
import type { Pack } from '../../index.js';

export function thai(): Pack {
  return {
    detectors: [thaiNationalIdDetector],
    generators: {
      first_name: firstName,
      last_name: lastName,
      person_name: fullName,
      national_id: nationalId,
      phone,
    },
    headers: HEADERS,
    titles: TITLES,
  };
}

// ─── Headers and titles ─────────────────────────────────────────

const HEADERS: Array<[RegExp, string | null]> = [
  [/^ค่า/, null],
  [/(?:ชื่อ\s*-?\s*(?:นาม)?สกุล|ชื่อพนักงาน|ชื่อลูกค้า)/, 'person_name'],
  [/นามสกุล/, 'last_name'],
  [/^(?:ชื่อ|ชื่อจริง)$/, 'first_name'],
  [/คำนำหน้า/, 'title'],
  [/รหัสพนักงาน/, 'employee_id'],
  [/(?:เลขบัตรประชาชน|เลขประจำตัวประชาชน|เลขที่บัตร)/, 'national_id'],
  [/อีเมล/, 'email'],
  [/(?:เบอร์|โทรศัพท์|มือถือ)/, 'phone'],
  [/ที่อยู่/, 'address'],
  [/เลขที่บัญชี/, 'bank_account'],
];

const TITLES: Array<[RegExp, 'male' | 'female']> = [
  // Longest first: นางสาว must not be read as นาง.
  [/^(?:นางสาว|นาง|น\.?ส\.?|ด\.?ญ\.?|เด็กหญิง)/, 'female'],
  [/^(?:นาย|ด\.?ช\.?|เด็กชาย|ว่าที่)/, 'male'],
];

// ─── Names ──────────────────────────────────────────────────────

const THAI_SCRIPT = /[฀-๿]/;

const firstName: Generator = (real, ctx) => {
  if (!THAI_SCRIPT.test(real)) return base.first_name(real, ctx);
  const male = (ctx.hints.sex ?? (ctx.rng() < 0.5 ? 'male' : 'female')) === 'male';
  if (ctx.attempt < 10) return pick(male ? FIRST_MALE : FIRST_FEMALE, ctx.rng);
  const front = male ? FRONT_MALE : FRONT_FEMALE;
  const end = pick(male ? END_MALE : END_FEMALE, ctx.rng);
  if (ctx.attempt < 100) return pick(front, ctx.rng) + end;
  // A whole workforce can use up two-part names: three parts, as long names are built.
  return pick(front, ctx.rng) + pick(front, ctx.rng) + end;
};

const lastName: Generator = (real, ctx) => {
  if (!THAI_SCRIPT.test(real)) return base.last_name(real, ctx);
  const root = pick(SURNAME_ROOTS, ctx.rng);
  if (ctx.attempt < 10) return root;
  if (ctx.attempt < 40) return root + pick(SURNAME_SUFFIXES, ctx.rng);
  if (ctx.attempt < 200) return root + pick(SURNAME_ROOTS, ctx.rng);
  // Thai surnames are near unique per family, so a large staff needs more than two roots give.
  return root + pick(SURNAME_ROOTS, ctx.rng) + pick(SURNAME_SUFFIXES, ctx.rng);
};

const fullName: Generator = (real, ctx) => `${firstName(real, ctx)} ${lastName(real, ctx)}`;

const FIRST_MALE = [
  'สมชาย', 'สมศักดิ์', 'ประเสริฐ', 'วิชัย', 'สุรชัย', 'ธนพล', 'ณัฐวุฒิ', 'กิตติพงษ์', 'อนุชา', 'ชัยวัฒน์',
  'ศักดิ์ชัย', 'ธีรวัฒน์', 'ปิยะพงษ์', 'วีรยุทธ', 'ภานุวัฒน์', 'สุทธิพงษ์', 'จักรพันธ์', 'ณรงค์', 'อภิชาติ', 'เอกชัย',
  'ธนากร', 'พงศกร', 'นพดล', 'ทวีศักดิ์', 'สมพงษ์', 'บุญมี', 'สมบัติ', 'วรวุฒิ', 'ศุภชัย', 'ปกรณ์',
  'กฤษดา', 'ธนวัฒน์', 'อดิศักดิ์', 'ไพโรจน์', 'เกียรติศักดิ์', 'ชาญวิทย์', 'ยุทธนา', 'ภูมิพัฒน์', 'ณัฐพล', 'พีรพัฒน์',
  'ธนภัทร', 'กิตติศักดิ์', 'สุริยา', 'ชนินทร์', 'วิทยา', 'อำนาจ', 'สุเมธ', 'อนันต์', 'ประพันธ์', 'เฉลิมพล',
  'ชาตรี', 'ธวัชชัย', 'พิษณุ', 'ราเชนทร์', 'วสันต์', 'ศราวุธ', 'อรรถพล', 'ทินกร', 'บรรจง', 'มานพ',
];

const FIRST_FEMALE = [
  'สมใจ', 'สุภาพร', 'วันเพ็ญ', 'กาญจนา', 'จันทร์เพ็ญ', 'ศิริพร', 'นภาพร', 'พรทิพย์', 'อรอุมา', 'ปวีณา',
  'กมลชนก', 'ณัฐธิดา', 'สุนิสา', 'วิไลวรรณ', 'ธิดารัตน์', 'เบญจวรรณ', 'รัตนา', 'อัญชลี', 'ศศิธร', 'จิราพร',
  'ปิยะนุช', 'มาลัย', 'สุดารัตน์', 'พัชรี', 'ชลธิชา', 'อารีรัตน์', 'นันทนา', 'วรรณา', 'เพ็ญศรี', 'ลัดดา',
  'กนกวรรณ', 'ธัญญารัตน์', 'พิมพ์ชนก', 'ภัทรา', 'สุพัตรา', 'จุฑามาศ', 'ดวงใจ', 'ประภาพร', 'อุไรวรรณ', 'ณัฐชา',
  'ชุติมา', 'วาสนา', 'สายสุนีย์', 'ทิพวรรณ', 'รุ่งนภา', 'อรทัย', 'ปาริชาติ', 'ยุพิน', 'สมพร', 'วิภาวดี',
  'เยาวลักษณ์', 'ศิริลักษณ์', 'สุกัญญา', 'อัมพร', 'บุษบา', 'จินตนา', 'ละเอียด', 'นิตยา', 'พจนีย์', 'ราตรี',
];

// Given names are built from parts; once the lists above run out, a fake is
// composed from a front and an end part, as real names are.
const FRONT_MALE = ['สม', 'ประ', 'วิ', 'สุ', 'ธน', 'ณัฐ', 'กิตติ', 'อนุ', 'ธีร', 'ปิยะ', 'วีร', 'ภานุ', 'อภิ', 'เอก', 'พง', 'ทวี', 'บุญ', 'วร', 'ศุภ', 'กฤษ', 'อดิ', 'เกียรติ', 'ชาญ', 'ภูมิ', 'พีร', 'ชนะ', 'ศรา', 'อรรถ', 'ทิน', 'เฉลิม'];
const END_MALE = ['ชาย', 'ศักดิ์', 'ชัย', 'พล', 'วุฒิ', 'พงษ์', 'วัฒน์', 'ยุทธ', 'ชาติ', 'กร', 'ดล', 'พันธ์', 'เดช', 'ภัทร', 'วิทย์', 'โชค', 'นันท์', 'พจน์'];
const FRONT_FEMALE = ['สม', 'สุ', 'วัน', 'กาญ', 'ศิริ', 'นภา', 'พร', 'อร', 'ปวี', 'กมล', 'ณัฐ', 'วิไล', 'ธิดา', 'เบญ', 'รัต', 'ศศิ', 'จิรา', 'ปิยะ', 'สุดา', 'ชล', 'อารี', 'นันท', 'เพ็ญ', 'กนก', 'ธัญ', 'พิมพ์', 'ภัท', 'จุฑา', 'ดวง', 'อุไร'];
const END_FEMALE = ['ใจ', 'พร', 'เพ็ญ', 'นา', 'ทิพย์', 'อุมา', 'วีณา', 'ชนก', 'ธิดา', 'นิสา', 'วรรณ', 'รัตน์', 'ลี', 'ธร', 'นุช', 'มาศ', 'ภา', 'ลักษณ์'];

const SURNAME_ROOTS = [
  'ใจดี', 'แสงทอง', 'ศรีสุข', 'บุญมา', 'ทองดี', 'รุ่งเรือง', 'สุขสวัสดิ์', 'มีสุข', 'พรหมมา', 'แก้วมณี',
  'จันทร์ดี', 'ศรีวงศ์', 'วงศ์ใหญ่', 'เพชรรัตน์', 'สมบูรณ์', 'ทองคำ', 'ชัยมงคล', 'บุญเรือง', 'แสงจันทร์', 'ประเสริฐสุข',
  'รัตนพันธ์', 'สุวรรณรัตน์', 'กิตติวงศ์', 'ธนสาร', 'พูลสวัสดิ์', 'เจริญผล', 'ศรีประเสริฐ', 'นาคสุข', 'อินทรสุข', 'ทรัพย์มาก',
  'พึ่งบุญ', 'ดวงแก้ว', 'มั่นคง', 'ศรีทอง', 'สายบุญ', 'ทองประเสริฐ', 'เรืองศรี', 'บุญประเสริฐ', 'ปัญญาดี', 'วัฒนกุล',
  'ธรรมรักษ์', 'มณีวงศ์', 'คงสมบูรณ์', 'แก้วประเสริฐ', 'ศรีสมบูรณ์', 'พันธุ์ดี', 'อ่อนศรี', 'ชื่นใจ', 'บุญยืน', 'สุขเจริญ',
  'ทองมา', 'ศรีเมือง', 'นิลสุวรรณ', 'หอมจันทร์', 'เกตุแก้ว', 'อุดมสุข', 'ประทุมทอง', 'จำปาทอง', 'สิทธิโชค', 'วงศ์สวัสดิ์',
];
const SURNAME_SUFFIXES = ['กุล', 'วงศ์', 'ศักดิ์', 'พันธ์', 'สกุล', 'ชัย', 'รักษ์', 'สิทธิ์'];

// ─── National ID ────────────────────────────────────────────────

/** A Thai national ID: 13 digits, optionally grouped 1-4-5-2-1, with a valid check digit. */
export function isThaiNationalId(value: string): boolean {
  const digits = value.replace(/[\s-]/g, '');
  if (!/^\d{13}$/.test(digits) || /[^\d\s-]/.test(value)) return false;
  const sum = [...digits.slice(0, 12)].reduce((s, d, i) => s + Number(d) * (13 - i), 0);
  return (11 - (sum % 11)) % 10 === Number(digits[12]);
}

const thaiNationalIdDetector: Detector = {
  type: 'national_id',
  detect(text) {
    const re = /\b\d[\s-]?\d{4}[\s-]?\d{5}[\s-]?\d{2}[\s-]?\d\b/g;
    const found = [];
    let m;
    while ((m = re.exec(text)) !== null) {
      if (isThaiNationalId(m[0])) found.push({ type: 'national_id', value: m[0], start: m.index, end: m.index + m[0].length });
    }
    return found;
  },
};

/** A Thai national ID with a valid check digit, in the same grouping; anything else keeps the base generator. */
const nationalId: Generator = (real, ctx) => {
  if (!isThaiNationalId(real)) return base.national_id(real, ctx);
  const body = [String(1 + Math.floor(ctx.rng() * 8)), ...Array.from({ length: 11 }, () => digit(ctx.rng))];
  const sum = body.reduce((s, d, i) => s + Number(d) * (13 - i), 0);
  const digits = [...body, String((11 - (sum % 11)) % 10)];
  let i = 0;
  return real.replace(/\d/g, () => digits[i++]!);
};

// ─── Phone ──────────────────────────────────────────────────────

/** Same format, separators kept, the network prefix (first two digits) kept, the rest new. */
const phone: Generator = (real, ctx) => {
  let seen = 0;
  return real.replace(/\d/g, (d) => (seen++ < 2 ? d : digit(ctx.rng)));
};

// ─── Helpers ────────────────────────────────────────────────────

function pick<T>(list: readonly T[], rng: () => number): T {
  return list[Math.floor(rng() * list.length)]!;
}

function digit(rng: () => number): string {
  return String(Math.floor(rng() * 10));
}
