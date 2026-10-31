# Dynamic intelligence level 2 experiment: registered plan

Registered on 2026-10-31, before the main run. Pass lines, arms and analysis below do not change after the main run starts. Any later change is reported as a deviation.

## Question

1. Does writing into an instance's weights add anything that writing into files does not?
2. Does an internalization loop add anything beyond automatic fine-tuning?

Definitions come from https://tenon.asia/blog/dynamic-intelligence/. This experiment tests level 2 only.

## Setup

- Base model: `Qwen/Qwen3-8B-MLX-4bit` at revision `383413e909f3bc5303ce195ebbdf0339c5a1a2a3`, Apache 2.0.
- Tooling: mlx-lm 0.32.2, Python 3.14, Apple M5 Pro with 64GB of unified memory.
- Task: synthetic expense audit for a fictional company, 10 interacting rules, scored by an exact rule engine. Decision, payable amount and approver must all be right.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, at most 3,072 output tokens. Unfinished answers count as wrong.

## Arms

All arms have the policy document, a concise-worksheet instruction, and notes the model rewrites each night from checked cases. The worksheet instruction is included because a single prompt line should not be credited to internalization.

| Arm | Writing | Level |
| --- | --- | --- |
| B | Files only | 0 |
| C | B plus a nightly LoRA on the day's checked logs | Weakest 2 |
| D | C plus practice: 50 new claims, 4 samples each, keep the shortest correct reasoning as a training target | Full 2 |

C and D use the same gradient steps: 200 iterations at batch size 2, up to 150 examples, learning rate 2e-5, 8 layers, rank 8. Half of the training examples have the policy removed. Three seeds per arm. One extra D instance learns Company Y, whose rules differ.

## Schedule

Twelve simulated days with 50 claims a day. Two rules change on day 10: the T2 hotel cap and the late-submission threshold. Old training examples that touch those rules are dropped from replay.

## Tests

Test claims are never used for training. Modifier pairs in the far set never appear in training or the near set.

| Point | Near with memory (100) | Far with memory (50) | Near cleared (100) | Near no-think (100) | General GSM8K (100) | Changed rules (50) |
| --- | --- | --- | --- | --- | --- | --- |
| Start, once for all arms | Yes | Yes | Yes | Yes | Yes | |
| Stable, after night 9 | Yes | Yes | Yes | Yes | Yes | |
| Changed day, new rules in memory, weights from night 9 | Yes | | | | | Yes |
| Recovered, after night 12 | Yes | | | | Yes | Yes |

The X and Y instances are also tested on both companies with external memory cleared, 50 claims each.

## Pass lines

Measured at the stable point on near claims, D against B, both with memory. All three must hold.

1. Median thinking tokens fall by at least 30%.
2. The 95% bootstrap interval for the accuracy difference has a lower bound of at least -3 percentage points.
3. General accuracy falls by at most 3 percentage points from the start point.

Internalization beats automatic fine-tuning if D's median thinking tokens are at least 20% below C's.

Secondary measures: cleared-memory accuracy, no-think accuracy, far-transfer accuracy, accuracy on changed rules at the changed day and recovered points, Company X and Y cross results, and the number of cases needed to pay back background tokens.

`report.py` applies these lines.

## Predictions

Written before the main run. These are judgments, not data.

| Measure | B | C | D |
| --- | --- | --- | --- |
| Near accuracy | 83–95% | Up 0–5 points | Down 3 to up 4 points |
| Thinking tokens against B | Baseline | Down 0–15% | Down 30–60% |
| Cleared-memory accuracy | 10–20% | 30–45% | 45–65% |
| Far accuracy against B | Baseline | Down 5 to up 3 points | Down 10 to up 2 points |
| General accuracy change | None | Down 0–2 points | Down 2–6 points |
| Drop on changed rules at the change | Near 0 | 10–25 points | 20–40 points |

| Claim | Probability |
| --- | --- |
| Thinking tokens fall by at least 30% | 65% |
| Accuracy line holds | 65% |
| General line holds | 55% |
| D beats C by at least 20% | 70% |
| All three pass lines hold | About 25% |

## How results will be read

| Result | Conclusion | Next step |
| --- | --- | --- |
| All three pass | Internalization adds a layer of ability over the harness | Pilot on a real workflow, then study level 1 |
| Only the general line fails | Internalization works but needs a gate | One more run with base-task replay |
| D is close to C | What we built is automatic fine-tuning | Publish the negative result and stop this route |
| Neither beats B | Level 0 is enough for this task | Publish the negative result and stop level 2 |

## Pilot changes

These happened before registration, on baseline runs only, and no adapter result informed them.

- 14 rules became 10, with the client-dinner and billable rules removed, to bring thinking down from about 2,600 tokens.
- The receipts and transport rules were reworded after most receipt cases were misread.
- The claim now states the total requested, and the rules explain that payable is often lower. Before this change a third of answers looped until the token cap.
- Each day was cut from 100 to 50 claims, and the test sets were resized, to fit about 65 hours of machine time.
- Pooled start accuracy on 48 baseline attempts is about 83%, within the 70–85% target. The start test will measure it on 100 claims.

## Limits

The task is synthetic, the check is perfect, and the model is small. Results do not transfer directly to real workflows or frontier models. The practice set comes from the rule engine, not from the model.
