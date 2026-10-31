import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dai.policy import (COMPANY_X, COMPANY_X_V2, COMPANY_Y, HELD_OUT, audit, is_correct, make_rng,
                        parse_answer, render_claim, render_policy, sample_claim)

BASE = {'id': 'C-1', 'grade': 'G3', 'dept': 'Engineering', 'tier': 'T2', 'nights': 2, 'days': 3,
        'rate': 120, 'meal_spend': 40, 'transport': 50, 'no_receipt': 0, 'violations': 0, 'days_late': 5}


class AuditTest(unittest.TestCase):
    def test_full_approval(self):
        self.assertEqual(audit(BASE, COMPANY_X),
                         {'decision': 'APPROVE', 'payable': 410, 'approver': 'Manager'})

    def test_interacting_rules(self):
        claim = dict(BASE, grade='G4', dept='Sales', tier='T1', nights=7, days=8, rate=260,
                     meal_spend=70, transport=500, no_receipt=130, violations=2, days_late=40)
        # Cap 220, reduced 176: 5*220 + 2*176 = 1452. Meals 8*60 = 480. Transport 400.
        # Subtotal 2332, minus 80 = 2252, then 90% twice: 2026, 1823.
        self.assertEqual(audit(claim, COMPANY_X),
                         {'decision': 'PARTIAL', 'payable': 1823, 'approver': 'VP'})

    def test_approver_bands(self):
        claim = dict(BASE, tier='T1', nights=3, days=4, rate=180, meal_spend=60, transport=100)
        self.assertEqual(audit(claim, COMPANY_X)['approver'], 'Manager')
        self.assertEqual(audit(dict(claim, nights=4, days=5), COMPANY_X)['approver'], 'Director')

    def test_late_reject(self):
        result = audit(dict(BASE, days_late=61), COMPANY_X)
        self.assertEqual(result, {'decision': 'REJECT', 'payable': 0, 'approver': 'None'})

    def test_policy_change_affects_result(self):
        claim = dict(BASE, rate=135, days_late=25)
        self.assertNotEqual(audit(claim, COMPANY_X), audit(claim, COMPANY_X_V2))


class GeneratorTest(unittest.TestCase):
    def test_seeded_generation_repeats(self):
        first = sample_claim(make_rng('x', 1), COMPANY_X)
        self.assertEqual(first, sample_claim(make_rng('x', 1), COMPANY_X))

    def test_held_out_pairs_stay_out_of_near_split(self):
        rng = make_rng('near')
        for _ in range(2000):
            modifiers = set(sample_claim(rng, COMPANY_X)['modifiers'])
            self.assertFalse(any(set(pair) <= modifiers for pair in HELD_OUT))

    def test_far_split_contains_one_held_out_pair(self):
        rng = make_rng('far')
        for _ in range(2000):
            modifiers = set(sample_claim(rng, COMPANY_X, 'far')['modifiers'])
            self.assertEqual(sum(set(pair) <= modifiers for pair in HELD_OUT), 1)

    def test_every_outcome_occurs_for_both_companies(self):
        for policy in (COMPANY_X, COMPANY_Y):
            rng = make_rng(policy.company)
            outcomes = {audit(sample_claim(rng, policy), policy)['decision'] for _ in range(1000)}
            self.assertEqual(outcomes, {'APPROVE', 'PARTIAL', 'REJECT'})

    def test_rendering_has_all_rules(self):
        self.assertEqual(render_policy(COMPANY_X).count('\nR'), 10)
        self.assertIn('Total requested $410.', render_claim(BASE))


class ParseTest(unittest.TestCase):
    TRUTH = {'decision': 'PARTIAL', 'payable': 100, 'approver': 'Manager'}

    def test_reads_last_json_after_thinking(self):
        text = '<think>{"decision": "REJECT", "payable": 0, "approver": "None"}</think>\n' \
               'Result: {"decision": "PARTIAL", "payable": 100, "approver": "Manager"}'
        self.assertTrue(is_correct(text, self.TRUTH))

    def test_unfinished_answer_is_wrong(self):
        self.assertIsNone(parse_answer('<think>still working'))
        self.assertFalse(is_correct('<think>{"decision": "PARTIAL", "payable": 100, "approver": "Manager"}', self.TRUTH))

    def test_no_think_mode_reads_plain_answer(self):
        self.assertTrue(is_correct('{"decision": "PARTIAL", "payable": 100, "approver": "Manager"}',
                                   self.TRUTH, thinking=False))


if __name__ == '__main__':
    unittest.main()
