# Dynamic intelligence

Status: the first experiment was stopped. No experiment is running.

The first registered experiment tested whether nightly fine-tuning makes a model better at a repeated synthetic expense-audit task. It was stopped on 2026-10-31, before any main-run result was read, because it measured skill on a repeated task and not memory of past interactions. Its plan and code remain at commit [cb696d9](https://github.com/dozybot001/dynamic-intelligence/tree/cb696d918d0ffe3a282ec44742266d9f8a23e0e3).

The current goal is a personal model in which every interaction affects the next one, with memory held in the model and not in its context. The next step is a benchmark for that kind of memory.
