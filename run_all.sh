#!/bin/bash
# Runs every arm in a fixed order. Finished steps are skipped, so it can restart at any time.
cd "$(dirname "$0")" || exit 1
for seed in 0 1 2; do
  for arm in B C D; do
    .venv/bin/python experiment.py --arm "$arm" --seed "$seed" || exit 1
  done
  if [ "$seed" = 0 ]; then
    .venv/bin/python experiment.py --arm D --company Y --seed 0 || exit 1
    .venv/bin/python experiment.py --cross || exit 1
  fi
done
.venv/bin/python report.py
