# Dynamic intelligence: level 2 experiment

This experiment tests whether writing to a model's weights while it runs adds anything beyond writing to files, and whether an internalization loop adds anything beyond automatic fine-tuning.

- Concept and levels: [Dynamic intelligence: models that change as they run](https://tenon.asia/blog/dynamic-intelligence/)
- Registered plan, pass lines and predictions: [PLAN.md](PLAN.md)
- Status: the main run is in progress. There are no results yet.

## Run it

It needs Apple silicon with about 32GB of unified memory and takes about 70 hours.

```bash
python3 -m venv .venv
.venv/bin/pip install mlx-lm==0.32.2
.venv/bin/hf download Qwen/Qwen3-8B-MLX-4bit --revision 383413e909f3bc5303ce195ebbdf0339c5a1a2a3 --local-dir models/qwen3-8b-4bit
mkdir -p data
curl -o data/gsm8k-test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl
./run_all.sh
.venv/bin/python report.py
```

Finished steps stay on disk, so `run_all.sh` can stop and restart at any time. It only runs on mains power.

`dai/policy.py` holds the synthetic task, its claim generator and the exact scoring rules. `tests/` checks them without a model.
