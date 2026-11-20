#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
mkdir -p logs
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
dataset="$HF_LEROBOT_HOME/$FH_REPO_ID"
if [[ -d "$dataset" && ! -f "$dataset/meta_fh.json" ]]; then
    backup=$(mktemp -d "$PI_WORK/incomplete-data.XXXXXX")
    mv -- "$dataset" "$backup/"
    printf 'Preserved incomplete conversion in %s\n' "$backup"
fi
bash prepare.sh > logs/conversion.log 2>&1
tail -4 logs/conversion.log

for cfg in pi05_fh_f0 pi05_fh_f2; do
    printf 'Computing %s normalization statistics\n' "$cfg"
    "$PI_PY" -u pi0_lora.py compute_norm_stats --config "$cfg" \
        --batch-size 32 --num-workers 4 > "logs/$cfg-norm.log" 2>&1
    tail -3 "logs/$cfg-norm.log"
done

"$PI_PY" -u pi0_lora.py verify-data > logs/verify-data.log 2>&1
tail -48 logs/verify-data.log
