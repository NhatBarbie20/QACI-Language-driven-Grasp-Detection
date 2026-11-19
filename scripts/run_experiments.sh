#!/usr/bin/env bash
# Train and evaluate every experiment of the paper, then print the results table.
#
#   bash scripts/run_experiments.sh                      # all four experiments
#   bash scripts/run_experiments.sh film film_cf         # a subset
#   EXTRA="train.epochs=10" bash scripts/run_experiments.sh
set -euo pipefail

experiments=("$@")
if [ ${#experiments[@]} -eq 0 ]; then
  experiments=(vision_only concat film film_cf)
fi

for name in "${experiments[@]}"; do
  python scripts/train.py --config "configs/${name}.yaml" --set ${EXTRA:-}
  python scripts/evaluate.py --run "outputs/${name}" --figure 8
done

python scripts/collect_results.py
