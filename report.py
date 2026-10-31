#!/usr/bin/env python3
"""Apply the registered pass lines in PLAN.md to whatever runs have finished."""
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent / 'runs'
SEEDS = (0, 1, 2)
BOOTSTRAP = 10000


def load(arm, seed, label):
    path = ROOT / f'{arm}-X-s{seed}' / 'eval' / f'{label}.json'
    return json.loads(path.read_text()) if path.exists() else None


def group(arm, label, part):
    rows = []
    for seed in SEEDS:
        report = load(arm, seed, label)
        if report:
            rows.append(report[part])
    return rows


def accuracy(rows):
    flat = [r for run in rows for r in run]
    return sum(r['correct'] for r in flat) / len(flat) if flat else None


def thinking(rows):
    flat = [r['thinking_tokens'] for run in rows for r in run]
    return statistics.median(flat) if flat else None


def accuracy_difference_ci(treated, baseline):
    """Bootstrap over test problems, keeping each problem's results across seeds together."""
    runs = min(len(treated), len(baseline))
    size = len(treated[0])
    rng = random.Random(0)
    differences = []
    for _ in range(BOOTSTRAP):
        picks = [rng.randrange(size) for _ in range(size)]
        total = sum(treated[s][i]['correct'] - baseline[s][i]['correct'] for s in range(runs) for i in picks)
        differences.append(total / (runs * size))
    differences.sort()
    return differences[int(BOOTSTRAP * 0.025)], differences[int(BOOTSTRAP * 0.975)]


def background_tokens(arm, seed):
    path = ROOT / f'{arm}-X-s{seed}' / 'metrics.jsonl'
    if not path.exists():
        return None
    total = 0
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get('step') in ('practice', 'train', 'notes') and event.get('day', 99) <= 9:
            total += event.get('tokens') or 0
    return total


def main():
    start = json.loads((ROOT / 'start-X.json').read_text()) if (ROOT / 'start-X.json').exists() else {}
    table = {}
    for arm in 'BCD':
        table[arm] = {
            'near accuracy': accuracy(group(arm, 'stable', 'near-memory')),
            'near thinking median': thinking(group(arm, 'stable', 'near-memory')),
            'far accuracy': accuracy(group(arm, 'stable', 'far-memory')),
            'cleared accuracy': accuracy(group(arm, 'stable', 'near-cleared')),
            'no-think accuracy': accuracy(group(arm, 'stable', 'near-nothink')),
            'general accuracy': accuracy(group(arm, 'stable', 'general')),
            'changed-day accuracy on changed rules': accuracy(group(arm, 'changed-day', 'changed-memory')),
            'recovered accuracy on changed rules': accuracy(group(arm, 'recovered', 'changed-memory')),
            'recovered general accuracy': accuracy(group(arm, 'recovered', 'general')),
            'runs': len(group(arm, 'stable', 'near-memory')),
        }
    print(json.dumps({'start': {k: round(accuracy([v]), 3) for k, v in start.items()}, 'arms': table}, indent=1))

    base, treated = group('B', 'stable', 'near-memory'), group('D', 'stable', 'near-memory')
    if not base or not treated:
        print('Primary comparison not available yet.')
        return
    reduction = 1 - thinking(treated) / thinking(base)
    low, high = accuracy_difference_ci(treated, base)
    general_drop = accuracy([start['general']]) - table['D']['general accuracy']
    versus_log = None
    log_rows = group('C', 'stable', 'near-memory')
    if log_rows:
        versus_log = 1 - thinking(treated) / thinking(log_rows)
    gates = {
        'thinking tokens down at least 30%': reduction >= 0.30,
        'accuracy difference CI lower bound at least -3 points': low >= -0.03,
        'general accuracy drop at most 3 points': general_drop <= 0.03,
    }
    saved = thinking(base) - thinking(treated)
    spent = statistics.mean(filter(None, (background_tokens('D', s) for s in SEEDS)))
    print(json.dumps({
        'thinking reduction D vs B': round(reduction, 3),
        'accuracy difference 95% CI': [round(low, 3), round(high, 3)],
        'general drop': round(general_drop, 3),
        'thinking reduction D vs C': None if versus_log is None else round(versus_log, 3),
        'internalization beats automatic fine-tuning': None if versus_log is None else versus_log >= 0.20,
        'cases to pay back background tokens': round(spent / saved) if saved > 0 else None,
        'gates': gates,
        'passed': all(gates.values()),
        'complete': len(treated) == len(SEEDS) and len(base) == len(SEEDS),
    }, indent=1))

    cross = ROOT / 'cross.json'
    if cross.exists():
        data = json.loads(cross.read_text())
        print(json.dumps({key: round(accuracy([rows]), 3) for key, rows in data.items()}, indent=1))


if __name__ == '__main__':
    main()
