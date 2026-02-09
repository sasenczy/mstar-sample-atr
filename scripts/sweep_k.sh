#!/usr/bin/env bash
# Sweep over k values (measured/synthetic mixing ratio) for the SAMPLE dataset.
# Trains one run per k value from 0.0 to 1.0 in increments of 0.05.
# All runs are logged to the 'sample-atr' W&B project.
#
# Usage:
#   bash scripts/sweep_k.sh
#   bash scripts/sweep_k.sh --dry-run    # Print commands without running

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

for i in $(seq 0 20); do
    k=$(python3 -c "print(f'{$i * 0.05:.2f}')")
    echo "=== Running k=${k} (${i}/20) ==="

    cmd=(
        python train.py
        dataset=sample
        dataset.k="${k}"
        logging.project=sample-atr
        logging.run_name="k_${k}"
        logging.tags="[sample,resnet18,k_sweep]"
    )

    if $DRY_RUN; then
        echo "  ${cmd[*]}"
    else
        "${cmd[@]}"
    fi
done

echo "Sweep complete."
