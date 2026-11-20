#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [[ -f demos.sha256 ]]; then
    (cd demos && sha256sum --quiet -c ../demos.sha256)
fi
run_variant() {
    local mode=$1 gpu=$2 port=$3
    local run="runs/paired/$mode"
    export CUDA_VISIBLE_DEVICES=$gpu
    if [[ ! -f "$run/checkpoint.pt" ]]; then bash run.sh train "$mode" "$run"; fi
    bash run.sh eval "$mode" "$run" 9100-9149 "$port" "${EVAL_WORKERS:-4}"
}
run_variant current "${CURRENT_GPU:-0}" 47181 > logs/current.log 2>&1 &
current=$!
run_variant history "${HISTORY_GPU:-1}" 47182 > logs/history.log 2>&1 &
history=$!
trap 'kill "$current" "$history" 2>/dev/null || true' INT TERM
result=0
wait "$current" || result=1
wait "$history" || result=1
if (( result != 0 )); then
    tail -n 24 logs/current.log logs/history.log
    exit "$result"
fi
bash run.sh compare | tee runs/paired/comparison.txt
