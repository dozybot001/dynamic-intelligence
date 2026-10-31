#!/usr/bin/env python3
"""Run one arm day by day. Finished steps stay on disk, so a run can stop and resume."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

from dai.llm import LLM, MODEL, ROOT
from dai.policy import (COMPANY_X, COMPANY_X_V2, COMPANY_Y, WORKSHEET, audit, is_correct, make_rng,
                        render_claim, render_policy, sample_claim)

ANSWER_FORMAT = 'Answer with one JSON line: {"decision": "...", "payable": 0, "approver": "..."}'
NOTES_SYSTEM = 'You keep short working notes for an expense auditor.'
GENERAL_SYSTEM = 'Solve the problem. End with one JSON line: {"answer": 0}'
FULL_EVAL = ('near-memory', 'far-memory', 'near-cleared', 'near-nothink', 'general')


@dataclass
class Config:
    arm: str = 'B'
    company: str = 'X'
    seed: int = 0
    days: int = 12
    change_day: int = 10
    tasks: int = 50
    practice: int = 50
    samples: int = 4
    examples: int = 150
    iters: int = 200
    train_batch: int = 2
    learning_rate: float = 2e-5
    num_layers: int = 8
    batch: int = 16
    max_tokens: int = 3072
    near: int = 100
    far: int = 50
    general: int = 100
    changed: int = 50

    @property
    def root(self):
        return ROOT / 'runs' / f'{self.arm}-{self.company}-s{self.seed}'


def wait_for_power():
    """Long runs only proceed on mains power."""
    while 'AC Power' not in subprocess.run(['pmset', '-g', 'ps'], capture_output=True, text=True).stdout:
        time.sleep(60)


def policy_for(cfg, day):
    if cfg.company == 'Y':
        return COMPANY_Y
    return COMPANY_X_V2 if day >= cfg.change_day else COMPANY_X


def memory_system(policy, notes):
    text = render_policy(policy) + '\n' + WORKSHEET
    return text + ('\nNotes from past cases:\n' + notes if notes else '')


def cleared_system(policy):
    return f'You are the expense auditor of {policy.company}. Apply its travel expense policy. {ANSWER_FORMAT}'


def claims(tag, count, policy, split='near'):
    rng = make_rng(tag)
    return [sample_claim(rng, policy, split) for _ in range(count)]


def changed_claims(tag, count):
    rng = make_rng(tag)
    found = []
    while len(found) < count:
        claim = sample_claim(rng, COMPANY_X_V2)
        if audit(claim, COMPANY_X) != audit(claim, COMPANY_X_V2):
            found.append(claim)
    return found


def solve(llm, cfg, items, system, policy, thinking, seed, repeat=1):
    prompts = [llm.prompt(system, render_claim(c), thinking) for c in items for _ in range(repeat)]
    results, elapsed, _ = llm.solve(prompts, cfg.max_tokens, cfg.batch, seed)
    rows = []
    for index, result in enumerate(results):
        claim = items[index // repeat]
        truth = audit(claim, policy)
        rows.append({**result, 'claim': claim, 'truth': truth,
                     'correct': is_correct(result['text'], truth, thinking)})
    return rows, elapsed


def log(cfg, **event):
    cfg.root.mkdir(parents=True, exist_ok=True)
    with (cfg.root / 'metrics.jsonl').open('a') as file:
        file.write(json.dumps({'time': time.time(), **event}) + '\n')


def update_notes(llm, cfg, policy, rows, notes, day):
    wrong = [r for r in rows if not r['correct'] and r['finished']][:6]
    right = sorted((r for r in rows if r['correct']), key=lambda r: r['thinking_tokens'])[:3]
    cases = [f"{render_claim(r['claim'])}\nCorrect result: {json.dumps(r['truth'])}" for r in wrong + right]
    request = (f"{render_policy(policy)}\n\nCurrent notes:\n{notes or '(none)'}\n\nRecent cases:\n"
               + '\n'.join(cases) + '\n\nRewrite the notes in at most 100 words. Keep shortcuts and '
               'common mistakes that help future cases. Return only the notes.')
    prompt = llm.prompt(NOTES_SYSTEM, request, thinking=False)
    results, elapsed, _ = llm.solve([prompt], 220, 1, cfg.seed + day)
    log(cfg, step='notes', day=day, tokens=results[0]['output_tokens'] + len(prompt), seconds=elapsed)
    return results[0]['text'].split('</think>')[-1].strip()


def example(cfg, rng, policy, system, row):
    chosen = system if rng.random() < 0.5 else cleared_system(policy)
    return {'messages': [{'role': 'system', 'content': chosen},
                         {'role': 'user', 'content': render_claim(row['claim'])},
                         {'role': 'assistant', 'content': row['text'].strip()}]}


def touches_changed_rule(claim):
    return claim['tier'] == 'T2' or claim['days_late'] > 20


def write_dataset(cfg, day, fresh, policy):
    rng = make_rng('dataset', cfg.arm, cfg.company, cfg.seed, day)
    pool = []
    for past in range(1, day):
        with (cfg.root / f'day-{past:02d}' / 'fresh.jsonl').open() as file:
            pool += [json.loads(line) for line in file]
    if cfg.company == 'X' and day >= cfg.change_day:
        pool = [item for item in pool if not touches_changed_rule(item['claim'])]
    chosen = fresh + rng.sample(pool, min(len(pool), max(0, cfg.examples - len(fresh))))
    rng.shuffle(chosen)
    data = cfg.root / f'day-{day:02d}' / 'data'
    data.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item['example']) for item in chosen[:cfg.examples]]
    (data / 'train.jsonl').write_text('\n'.join(lines) + '\n')
    (data / 'valid.jsonl').write_text('\n'.join(lines[:cfg.train_batch]) + '\n')
    (cfg.root / f'day-{day:02d}' / 'fresh.jsonl').write_text(
        ''.join(json.dumps(item) + '\n' for item in fresh))
    return data, len(lines)


def train(cfg, day, data, previous):
    target = cfg.root / f'day-{day:02d}' / 'adapter'
    command = [str(ROOT / '.venv/bin/mlx_lm.lora'), '--model', MODEL, '--train', '--data', str(data),
               '--adapter-path', str(target), '--fine-tune-type', 'lora', '--mask-prompt',
               '--num-layers', str(cfg.num_layers), '--batch-size', str(cfg.train_batch),
               '--iters', str(cfg.iters), '--learning-rate', str(cfg.learning_rate),
               '--max-seq-length', str(cfg.max_tokens + 700), '--grad-checkpoint',
               '--seed', str(cfg.seed + day), '--val-batches', '1', '--steps-per-eval', '100000',
               '--steps-per-report', '50', '--save-every', '100000']
    if previous:
        command += ['--resume-adapter-file', str(Path(previous) / 'adapters.safetensors')]
    start = time.time()
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    seconds = time.time() - start
    (target / 'train.log').write_text(output)
    plain = re.sub(r'\x1b\[[0-9;]*m', '', output)
    totals = re.findall(r'([\d.]+)k\s*$', plain, re.M)
    log(cfg, step='train', day=day, seconds=seconds, iters=cfg.iters,
        tokens=int(float(totals[-1]) * 1000) if totals else None)
    return str(target)


def adapter_after(cfg, day):
    if cfg.arm == 'B' or day < 1:
        return None
    return str(cfg.root / f'day-{day:02d}' / 'adapter')


def notes_after(cfg, day):
    path = cfg.root / f'day-{day:02d}' / 'notes.md'
    return path.read_text() if day >= 1 and path.exists() else ''


def run_day(cfg, day):
    folder = cfg.root / f'day-{day:02d}'
    if (folder / 'done').exists():
        return
    wait_for_power()
    folder.mkdir(parents=True, exist_ok=True)
    policy = policy_for(cfg, day)
    previous = adapter_after(cfg, day - 1)
    notes = notes_after(cfg, day - 1)
    system = memory_system(policy, notes)
    rng = make_rng('assign', cfg.arm, cfg.company, cfg.seed, day)
    llm = LLM(adapter=previous)

    items = claims(f'{cfg.company}|{cfg.seed}|day|{day}', cfg.tasks, policy)
    rows, elapsed = solve(llm, cfg, items, system, policy, True, cfg.seed * 1000 + day)
    log(cfg, step='tasks', day=day, seconds=elapsed, tokens=sum(r['output_tokens'] for r in rows),
        accuracy=sum(r['correct'] for r in rows) / len(rows))
    (folder / 'tasks.json').write_text(json.dumps(rows))
    (folder / 'notes.md').write_text(update_notes(llm, cfg, policy, rows, notes, day))

    if cfg.arm != 'B':
        verified = [r for r in rows if r['correct']]
        if cfg.arm == 'D':
            practice = claims(f'{cfg.company}|{cfg.seed}|practice|{day}', cfg.practice, policy)
            tries, elapsed = solve(llm, cfg, practice, system, policy, True, cfg.seed * 2000 + day, cfg.samples)
            log(cfg, step='practice', day=day, seconds=elapsed, tokens=sum(r['output_tokens'] for r in tries),
                accuracy=sum(r['correct'] for r in tries) / len(tries))
            for index in range(cfg.practice):
                group = [r for r in tries[index * cfg.samples:(index + 1) * cfg.samples] if r['correct']]
                if group:
                    verified.append(min(group, key=lambda r: r['thinking_tokens']))
        fresh = [{'claim': r['claim'], 'example': example(cfg, rng, policy, system, r)} for r in verified]
        del llm
        data, size = write_dataset(cfg, day, fresh, policy)
        log(cfg, step='dataset', day=day, fresh=len(fresh), examples=size)
        train(cfg, day, data, previous)
    (folder / 'done').write_text('')


def score_general(text):
    body = text.rsplit('</think>', 1)[-1] if '</think>' in text else ''
    found = re.findall(r'-?\d[\d,]*\.?\d*', body)
    return float(found[-1].replace(',', '').rstrip('.')) if found else None


def general_items(count):
    items = []
    for line in (ROOT / 'data' / 'gsm8k-test.jsonl').read_text().splitlines()[:count]:
        record = json.loads(line)
        answer = float(record['answer'].split('####')[-1].strip().replace(',', ''))
        items.append((record['question'], answer))
    return items


def evaluate(cfg, label, after_day, policy, parts=FULL_EVAL):
    # Before any writing every arm is the same model, so the start test runs once per company.
    target = ROOT / 'runs' / f'start-{cfg.company}.json' if label == 'start' \
        else cfg.root / 'eval' / f'{label}.json'
    if target.exists():
        return
    wait_for_power()
    target.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(adapter=adapter_after(cfg, after_day))
    notes = notes_after(cfg, after_day)
    memory = memory_system(policy, notes)
    report = {}
    near = claims(f'{cfg.company}|test-near', cfg.near, policy)
    if 'near-memory' in parts:
        report['near-memory'], _ = solve(llm, cfg, near, memory, policy, True, 7)
    if 'far-memory' in parts:
        far = claims(f'{cfg.company}|test-far', cfg.far, policy, 'far')
        report['far-memory'], _ = solve(llm, cfg, far, memory, policy, True, 7)
    if 'near-cleared' in parts:
        report['near-cleared'], _ = solve(llm, cfg, near, cleared_system(policy), policy, True, 8)
    if 'near-nothink' in parts:
        report['near-nothink'], _ = solve(llm, cfg, near, memory, policy, False, 9)
    if 'changed-memory' in parts:
        items = changed_claims('X|test-changed', cfg.changed)
        report['changed-memory'], _ = solve(llm, cfg, items, memory, policy, True, 10)
    if 'general' in parts:
        problems = general_items(cfg.general)
        prompts = [llm.prompt(GENERAL_SYSTEM, question) for question, _ in problems]
        results, _, _ = llm.solve(prompts, cfg.max_tokens, cfg.batch, 11)
        report['general'] = [{**r, 'correct': score_general(r['text']) == answer}
                             for r, (_, answer) in zip(results, problems)]
    target.write_text(json.dumps(report))
    summary = {name: {'accuracy': round(sum(r['correct'] for r in rows) / len(rows), 3),
                      'thinking_median': statistics.median(r['thinking_tokens'] for r in rows),
                      'finished': round(sum(r['finished'] for r in rows) / len(rows), 3)}
               for name, rows in report.items()}
    log(cfg, step='eval', label=label, summary=summary)
    print(label, json.dumps(summary), flush=True)


def cross(cfg):
    """Test the X and Y instances on both companies with external memory cleared."""
    target = ROOT / 'runs' / 'cross.json'
    if target.exists():
        return
    wait_for_power()
    adapters = {'X': ROOT / 'runs' / 'D-X-s0' / 'day-09' / 'adapter',
                'Y': ROOT / 'runs' / 'D-Y-s0' / 'day-12' / 'adapter'}
    policies = {'X': COMPANY_X, 'Y': COMPANY_Y}
    report = {}
    for instance, adapter in adapters.items():
        llm = LLM(adapter=str(adapter))
        for company, policy in policies.items():
            items = claims(f'{company}|test-cross', 50, policy)
            rows, _ = solve(llm, cfg, items, cleared_system(policy), policy, True, 12)
            report[f'{instance}-on-{company}'] = rows
        del llm
    target.write_text(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    for name, value in asdict(Config()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
    parser.add_argument('--until', type=int, help='Stop after this day')
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--cross', action='store_true')
    args = vars(parser.parse_args())
    until, no_eval, run_cross = args.pop('until'), args.pop('no_eval'), args.pop('cross')
    cfg = Config(**args)
    if run_cross:
        cross(cfg)
        return
    cfg.root.mkdir(parents=True, exist_ok=True)
    (cfg.root / 'config.json').write_text(json.dumps(asdict(cfg), indent=1))
    last = min(until or cfg.days, cfg.days)
    stable = cfg.change_day - 1 if cfg.company == 'X' else cfg.days

    if not no_eval:
        evaluate(cfg, 'start', 0, policy_for(cfg, 1))
    for day in range(1, last + 1):
        if day == stable + 1 and not no_eval:
            evaluate(cfg, 'stable', stable, policy_for(cfg, stable))
            if cfg.company == 'X':
                evaluate(cfg, 'changed-day', stable, COMPANY_X_V2, ('near-memory', 'changed-memory'))
        run_day(cfg, day)
    if not no_eval and last == cfg.days:
        if cfg.company == 'Y':
            evaluate(cfg, 'stable', last, COMPANY_Y)
        else:
            evaluate(cfg, 'recovered', last, COMPANY_X_V2, ('near-memory', 'changed-memory', 'general'))


if __name__ == '__main__':
    main()
