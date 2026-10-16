/**
 * HR pack — payroll and personnel files: employee codes, the people who
 * prepared and signed off a run, and the details an analysis never needs.
 *
 *   import { PrivacyProxy } from 'pii-proxy';
 *   import { hr } from 'pii-proxy/packs/hr';
 *   const proxy = new PrivacyProxy({ use: [hr()] });
 *
 * A domain pack owns rules, not words: which columns identify someone in this
 * field and how each is treated. The words for them in another language come
 * from that locale's pack (`packs/th` knows รหัสพนักงาน), so the two combine:
 * `use: [thai(), hr()]`.
 *
 * Addresses and bank accounts are withheld outright — `[address withheld #3]`
 * — rather than faked: nothing computed from a payroll needs them, and a
 * plausible fake account number is one more thing someone might act on.
 */
import { withheld } from '../../generators/index.js';
import type { Pack } from '../../index.js';

export function hr(): Pack {
  return {
    headers: HEADERS,
    generators: {
      address: withheld('address'),
      bank_account: withheld('bank account'),
    },
  };
}

/** Matched against the header squashed: "Emp Code" → "empcode", "Prepared By" → "preparedby". */
const HEADERS: Array<[RegExp, string | null]> = [
  [/^(?:emp|employee|staff|personnel|worker)(?:code|id|no|number|num)$/, 'employee_id'],
  [/^(?:emp)?(?:temp)?cardno$/, 'employee_id'],
  [/^(?:reportto|manager|supervisor)(?:emp)?(?:code|id|no)$/, 'employee_id'],
  [/^(?:reportto(?:emp)?|manager|supervisor|head)(?:full)?name(?:eng|en|th)?$/, 'person_name'],
  [/^(?:approver|approvedby|auditor|organizer|organiser|reviewer|checker|requester|requestedby|preparedby|createdby)(?:eng|en|th)?$/, 'person_name'],
  [/^workpermit(?:no|number)?$/, 'national_id'],
];
