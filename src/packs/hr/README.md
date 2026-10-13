# HR pack — `pii-proxy/packs/hr`

Payroll and personnel files for [pii-proxy](../../../README.md): employee codes, the people who prepared and signed off a run, and the details an analysis never needs.

```typescript
import { PrivacyProxy } from 'pii-proxy';
import { hr } from 'pii-proxy/packs/hr';
import { maskFiles } from 'pii-proxy/files';

const proxy = new PrivacyProxy({ use: [hr()] });
const { results } = await maskFiles([{ name: 'payroll-09.xlsx', bytes }], proxy);
```

## What it adds

| Columns | Type | Fake |
|---|---|---|
| `EmpCode`, `EmployeeId`, `StaffNo`, … | `employee_id` | Letters and first two digits kept (often a hire year), the rest new |
| `Approver`, `Auditor`, `Organizer`, `Prepared By`, `Reviewer`, … | `person_name` | A fake name — whoever signed off a run is a person too |
| Address | `address` | Withheld: `[address withheld #3]` |
| Bank account | `bank_account` | Withheld: `[bank account withheld #12]` |

Addresses and bank accounts are withheld rather than faked: nothing computed from a payroll needs them, and a plausible fake account number is one more thing someone might act on.

## With a locale

A domain pack owns rules, not words. The words for these columns in another language come from that locale's pack — [`packs/th`](../th/README.md) knows รหัสพนักงาน — so they combine:

```typescript
const proxy = new PrivacyProxy({ use: [thai(), hr()] });
```
