# Thai pack — `pii-proxy/packs/th`

Thai names, national IDs and phone numbers for [pii-proxy](../../../README.md), plus the Thai column headers and titles that [`pii-proxy/files`](../../../README.md#masking-files-pii-proxyfiles) needs to mask a Thai spreadsheet.

```typescript
import { PrivacyProxy } from 'pii-proxy';
import { thai } from 'pii-proxy/packs/th';
import { maskFiles } from 'pii-proxy/files';

const proxy = new PrivacyProxy({ use: [thai()] });
const { results } = await maskFiles([{ name: 'เงินเดือน 09.2569.xlsx', bytes }], proxy);
```

## What it adds

| | Detection | Fake |
|---|---|---|
| Name (first, last, full) | Table columns | A Thai name for a Thai name, an English one for a name in Latin script — so a roster with both stays consistent. Follows the title's sex. |
| National ID | 13 digits (grouped 1-4-5-2-1 or not) with a valid check digit | Valid check digit, same grouping |
| Phone | Core phone detector | Same format, network prefix (first two digits) kept |

**Names.** The lists cover a few hundred people; past that, a fake is composed from a front and an end part (สม + ชัย, ศรี + สุข), the way Thai names are built, so a large roster never falls back to codes or hyphenated pairs.

**Column headers** for `pii-proxy/files`:

| Header | Type |
|---|---|
| ชื่อ, ชื่อจริง | `first_name` |
| นามสกุล | `last_name` |
| ชื่อ-นามสกุล, ชื่อพนักงาน, ชื่อลูกค้า | `person_name` |
| คำนำหน้า | `title` |
| รหัสพนักงาน | `employee_id` |
| เลขบัตรประชาชน, เลขประจำตัวประชาชน | `national_id` |
| อีเมล | `email` |
| เบอร์, โทรศัพท์, มือถือ | `phone` |
| ที่อยู่ | `address` |
| เลขที่บัญชี | `bank_account` |

A header starting with ค่า names a fee or an allowance (ค่าโทร is a phone allowance, an amount), so it is never taken for a contact detail. A leading item number (`14.ค่าโทรฯ`) is ignored.

**Titles.** นาย, ด.ช., เด็กชาย and ว่าที่ give a male fake; นาง, นางสาว, น.ส., ด.ญ. and เด็กหญิง a female one — next to the core's Mr / Mrs / Miss / Ms.

## Example

| รหัสพนักงาน | คำนำหน้า | ชื่อ | นามสกุล | ชื่อ-นามสกุล | แผนก | เงินเดือน |
|---|---|---|---|---|---|---|
| EMP68035 | นาย | ทดสอบชัย | ทองแท้ | นายทดสอบชัย ทองแท้ | คลังสินค้า | 21465 |
| ↓ | | | | | | |
| EMP68417 | นาย | ธนพล | ศรีวงศ์ | นายธนพล ศรีวงศ์ | คลังสินค้า | 21465 |

The code keeps its letters and first two digits (often a hire year), the full name is rebuilt from the masked parts, and the department and salary are untouched.
