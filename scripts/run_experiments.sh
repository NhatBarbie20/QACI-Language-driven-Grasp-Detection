#!/usr/bin/env bash
# Train and evaluate the experiments of the paper, then print the results table.
#
#   bash scripts/run_experiments.sh                      # baselines and the proposed model
#   bash scripts/run_experiments.sh film factor_all      # a subset
#   EXTRA="train.epochs=10" bash scripts/run_experiments.sh
set -euo pipefail

experiments=("$@")
if [ ${#experiments[@]} -eq 0 ]; then
  experiments=(vision_only concat film factor_all)
fi

for name in "${experiments[@]}"; do
  python scripts/train.py --config "configs/${name}.yaml" --set ${EXTRA:-}
  python scripts/evaluate.py --run "outputs/${name}" --figure 8
  # prompt-swap metrics on label-disjoint object pairs
  python scripts/evaluate.py --run "outputs/${name}" --max-negative-overlap 0.1
  if [[ "${name}" == factor* ]]; then  # geometry factor alone, and ground-truth grounding
    python scripts/evaluate.py --run "outputs/${name}" --grounding none
    python scripts/evaluate.py --run "outputs/${name}" --grounding oracle
  fi
done

python scripts/collect_results.py
