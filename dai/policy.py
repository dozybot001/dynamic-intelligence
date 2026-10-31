"""Synthetic expense-audit domain: policies, claim generator and exact oracle."""
from dataclasses import dataclass, field, replace
import json
import random
import re

GRADES = ('G1', 'G2', 'G3', 'G4', 'G5')
DEPARTMENTS = ('Sales', 'Engineering', 'Finance', 'Operations')
TIERS = ('T1', 'T2', 'T3')
MODIFIERS = ('senior', 'long_stay', 'sales', 'receipts', 'violations', 'late')
# These pairs never appear in training or near-transfer tests.
HELD_OUT = (('long_stay', 'violations'), ('senior', 'late'))
WORKSHEET = ('Work through R1 to R10 once, as short lines of arithmetic. '
             'Do not repeat yourself or re-check finished steps.')


@dataclass(frozen=True)
class Policy:
    company: str
    hotel_cap: dict = field(default_factory=lambda: {'T1': 180, 'T2': 140, 'T3': 100})
    senior_uplift: int = 40
    long_stay_after: int = 5
    long_stay_pct: int = 80
    meal_cap: dict = field(default_factory=lambda: {'T1': 60, 'T2': 50, 'T3': 40})
    transport_cap: int = 30
    sales_transport_cap: int = 50
    receipt_allowance: int = 50
    violation_threshold: int = 2
    violation_pct: int = 90
    late_after: int = 30
    late_pct: int = 90
    reject_after: int = 60
    director_from: int = 1000
    vp_from: int = 3000


COMPANY_X = Policy('Company X')
COMPANY_Y = Policy(
    'Company Y',
    hotel_cap={'T1': 210, 'T2': 150, 'T3': 90}, senior_uplift=25, long_stay_after=4,
    long_stay_pct=70, meal_cap={'T1': 70, 'T2': 45, 'T3': 35}, transport_cap=25,
    sales_transport_cap=60, receipt_allowance=30, violation_threshold=3, violation_pct=85,
    late_after=21, late_pct=80, reject_after=45, director_from=800, vp_from=2500,
)
# Two rules change on day 10.
COMPANY_X_V2 = replace(COMPANY_X, late_after=20, hotel_cap={'T1': 180, 'T2': 120, 'T3': 100})


def render_policy(p):
    return f"""{p.company} travel expense policy. Apply the rules in this order.
R1 Hotel cap per night: T1 ${p.hotel_cap['T1']}, T2 ${p.hotel_cap['T2']}, T3 ${p.hotel_cap['T3']}. Each night pays the lower of the rate and the cap.
R2 Grades G4 and G5 add ${p.senior_uplift} to the hotel cap.
R3 Nights after night {p.long_stay_after} use {p.long_stay_pct}% of the cap, rounded down to whole dollars.
R4 Meal cap per day: T1 ${p.meal_cap['T1']}, T2 ${p.meal_cap['T2']}, T3 ${p.meal_cap['T3']}. Each day pays the lower of the daily spend and the cap.
R5 Transport limit is ${p.transport_cap} times trip days, or ${p.sales_transport_cap} times trip days for Sales staff. Transport pays the lower of the local transport total and this limit.
R6 Subtotal is hotel plus meals plus transport. The first ${p.receipt_allowance} without receipts is allowed. Deduct only the excess above ${p.receipt_allowance} from the subtotal.
R7 With {p.violation_threshold} or more violations in the last 12 months, multiply by {p.violation_pct}%, rounded down.
R8 Submitted more than {p.late_after} days after the trip, multiply by {p.late_pct}%, rounded down. More than {p.reject_after} days after, pay $0.
R9 Total requested is the employee's request before these rules, so payable is often lower. Decision: APPROVE if payable equals total requested, REJECT if payable is $0, otherwise PARTIAL.
R10 Approver: None for REJECT. Otherwise VP from ${p.vp_from} payable, Director from ${p.director_from}, else Manager. Grades G4 and G5 always need VP.
Answer with one JSON line: {{"decision": "...", "payable": 0, "approver": "..."}}"""


def claimed_total(c):
    return c['nights'] * c['rate'] + c['days'] * c['meal_spend'] + c['transport']


def audit(c, p):
    cap = p.hotel_cap[c['tier']]
    if c['grade'] in ('G4', 'G5'):
        cap += p.senior_uplift
    reduced = cap * p.long_stay_pct // 100
    normal = min(c['nights'], p.long_stay_after)
    hotel = normal * min(c['rate'], cap) + (c['nights'] - normal) * min(c['rate'], reduced)
    meals = c['days'] * min(c['meal_spend'], p.meal_cap[c['tier']])
    per_day = p.sales_transport_cap if c['dept'] == 'Sales' else p.transport_cap
    transport = min(c['transport'], c['days'] * per_day)

    amount = max(0, hotel + meals + transport - max(0, c['no_receipt'] - p.receipt_allowance))
    if c['violations'] >= p.violation_threshold:
        amount = amount * p.violation_pct // 100
    if c['days_late'] > p.reject_after:
        amount = 0
    elif c['days_late'] > p.late_after:
        amount = amount * p.late_pct // 100

    decision = 'REJECT' if amount == 0 else 'APPROVE' if amount == claimed_total(c) else 'PARTIAL'
    if decision == 'REJECT':
        approver = 'None'
    elif c['grade'] in ('G4', 'G5') or amount >= p.vp_from:
        approver = 'VP'
    else:
        approver = 'Director' if amount >= p.director_from else 'Manager'
    return {'decision': decision, 'payable': amount, 'approver': approver}


def render_claim(c):
    return (f"Claim {c['id']}. Grade {c['grade']}, {c['dept']} department. Trip to a {c['tier']} city: "
            f"{c['nights']} nights, {c['days']} days. Hotel ${c['rate']} per night. "
            f"Meals ${c['meal_spend']} per day. Local transport total ${c['transport']}. "
            f"Total requested ${claimed_total(c)}. Amount without receipts ${c['no_receipt']}. "
            f"Violations in the last 12 months: {c['violations']}. "
            f"Submitted {c['days_late']} days after the trip.")


def _pick_modifiers(rng, split, size):
    pairs = [set(pair) for pair in HELD_OUT]
    while True:
        chosen = set(rng.sample(MODIFIERS, size))
        held = sum(pair <= chosen for pair in pairs)
        if (split == 'far' and held == 1) or (split != 'far' and held == 0):
            return chosen


def sample_claim(rng, policy, split='near', size=None, claim_id=None):
    size = size or rng.choice((1, 2, 2, 3))
    if split == 'far':
        size = max(size, 2)
    m = _pick_modifiers(rng, split, size)
    tier = rng.choice(TIERS)
    nights = rng.randint(policy.long_stay_after + 1, policy.long_stay_after + 3) if 'long_stay' in m \
        else rng.randint(1, policy.long_stay_after)
    hotel_cap = policy.hotel_cap[tier]
    meal_cap = policy.meal_cap[tier]
    c = {
        'id': claim_id or f'C-{rng.randint(1000, 9999)}',
        'grade': rng.choice(('G4', 'G5')) if 'senior' in m else rng.choice(('G1', 'G2', 'G3')),
        'dept': 'Sales' if 'sales' in m else rng.choice(DEPARTMENTS[1:]),
        'tier': tier, 'nights': nights, 'days': nights + 1,
        'rate': rng.randint(hotel_cap - 40, hotel_cap + 80),
        'meal_spend': rng.randint(meal_cap - 15, meal_cap + 20),
        'no_receipt': rng.randint(policy.receipt_allowance + 10, policy.receipt_allowance + 150)
        if 'receipts' in m else rng.randint(0, policy.receipt_allowance),
        'violations': rng.randint(policy.violation_threshold, policy.violation_threshold + 2)
        if 'violations' in m else rng.randint(0, policy.violation_threshold - 1),
        'days_late': rng.randint(policy.reject_after + 1, policy.reject_after + 30)
        if 'late' in m and rng.random() < 0.12
        else rng.randint(policy.late_after + 1, policy.reject_after)
        if 'late' in m else rng.randint(1, policy.late_after),
    }
    per_day = policy.sales_transport_cap if c['dept'] == 'Sales' else policy.transport_cap
    c['transport'] = rng.randint(max(10, c['days'] * per_day - 60), c['days'] * per_day + 80)
    # Some claims sit under every cap so that a full approval stays a real answer.
    if rng.random() < 0.2:
        c['rate'] = rng.randint(hotel_cap - 40, hotel_cap)
        c['meal_spend'] = rng.randint(meal_cap - 15, meal_cap)
        c['transport'] = rng.randint(10, c['days'] * per_day)
    c['modifiers'] = sorted(m)
    return c


def parse_answer(text, thinking=True):
    """Return the last JSON object after the thinking block, or None."""
    if thinking and '</think>' not in text:
        return None
    body = text.rsplit('</think>', 1)[-1]
    for match in reversed(re.findall(r'\{[^{}]*\}', body)):
        try:
            data = json.loads(match)
            return {'decision': str(data['decision']).upper(), 'payable': int(data['payable']),
                    'approver': str(data['approver'])}
        except (ValueError, KeyError, TypeError):
            continue
    return None


def is_correct(text, truth, thinking=True):
    return parse_answer(text, thinking) == truth


def make_rng(*parts):
    return random.Random('|'.join(str(part) for part in parts))
