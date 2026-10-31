#!/usr/bin/env python3
"""Calibrate the baseline: accuracy, thinking tokens and throughput with the policy in context."""
import argparse
import json
from pathlib import Path
import statistics

from dai.llm import LLM, ROOT
from dai.policy import COMPANY_X, WORKSHEET, audit, is_correct, make_rng, render_claim, render_policy, sample_claim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=1536)
    parser.add_argument('--no-think', action='store_true')
    parser.add_argument('--no-policy', action='store_true')
    parser.add_argument('--worksheet', action='store_true')
    parser.add_argument('--adapter')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    llm = LLM(adapter=args.adapter)
    rng = make_rng('pilot', args.seed)
    claims = [sample_claim(rng, COMPANY_X) for _ in range(args.n)]
    system = 'You are an expense auditor. Answer with one JSON line.' if args.no_policy \
        else render_policy(COMPANY_X)
    if args.worksheet:
        system += '\n' + WORKSHEET
    thinking = not args.no_think
    prompts = [llm.prompt(system, render_claim(c), thinking) for c in claims]
    results, elapsed, stats = llm.solve(prompts, args.max_tokens, args.batch, args.seed)

    rows = []
    for claim, result in zip(claims, results):
        truth = audit(claim, COMPANY_X)
        rows.append({**result, 'truth': truth, 'correct': is_correct(result['text'], truth, thinking),
                     'modifiers': claim['modifiers']})
    out = ROOT / 'runs' / 'pilot'
    out.mkdir(parents=True, exist_ok=True)
    label = f"n{args.n}-b{args.batch}-{'nothink' if args.no_think else 'think'}-{'nodoc' if args.no_policy else 'doc'}{'-ws' if args.worksheet else ''}"
    (out / f'{label}.json').write_text(json.dumps(rows, indent=1))

    output_tokens = sum(r['output_tokens'] for r in rows)
    thinking_tokens = sorted(r['thinking_tokens'] for r in rows)
    print(json.dumps({
        'label': label,
        'accuracy': round(sum(r['correct'] for r in rows) / len(rows), 3),
        'finished': round(sum(r['finished'] for r in rows) / len(rows), 3),
        'thinking_median': statistics.median(thinking_tokens),
        'thinking_p90': thinking_tokens[int(len(thinking_tokens) * 0.9)],
        'prompt_tokens': rows[0]['prompt_tokens'],
        'seconds': round(elapsed, 1),
        'output_tok_per_s': round(output_tokens / elapsed, 1),
        'prompt_tps': round(stats.prompt_tps, 1),
        'generation_tps': round(stats.generation_tps, 1),
        'peak_gb': round(stats.peak_memory, 1),
    }, indent=1))


if __name__ == '__main__':
    main()
