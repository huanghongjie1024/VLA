#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
mkdir -p logs
for cfg in pi05_fh_f0 pi05_fh_f2; do
    test -f "assets/$cfg/$FH_REPO_ID/norm_stats.json"
    printf '[pi0] Starting %s for 10000 steps on GPUs 0 and 1\n' "$cfg"
    CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.75         "$PI_PY" -u pi0_lora.py train         --config "$cfg" --exp-name peg_lora_10k_dual --fsdp 1         --steps 10000 --batch-size 8 --warmup 500 --seed 42         --action-horizon 10 --num-workers 4         --save-interval 5000 --keep-period 5000         2>&1 | tee "logs/$cfg-10k-dual.log"
done
