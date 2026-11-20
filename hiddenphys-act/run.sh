#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
python_exec=${HIDDENPHYS_PYTHON:-python}
export PYTHONPATH="$root/sim_packages:$root/ForceHarness:${LEROBOT_ROOT:-$root/..}/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
stage=${1:?smoke, train, eval, or compare}
mode=${2:-current}
case "$mode" in current|history) ;; *) exit 2 ;; esac
case "$stage" in
    smoke)
        run="$root/runs/smoke"
        if [[ ! -f "$run/checkpoint.pt" ]]; then
            "$python_exec" -u experiment.py train --mode current --out "$run" --limit 8 --steps 12 --batch 8 --workers 2
        fi
        EVAL_DIR=${EVAL_DIR:-"$run/eval_pinned"} bash run.sh eval current "$run" 9100-9100 47181 1
        ;;
    train)
        run=${3:-"$root/runs/paired/$mode"}
        "$python_exec" -u experiment.py train --mode "$mode" --out "$run" --steps "${STEPS:-5000}" --batch "${BATCH:-32}" --workers "${DATA_WORKERS:-6}" --imagenet
        ;;
    eval)
        run=${3:-"$root/runs/paired/$mode"}
        seeds=${4:-9100-9149}
        port=${5:-47181}
        workers=${6:-4}
        eval_dir=${EVAL_DIR:-"$run/eval"}
        eval_dir=$(realpath -m "$eval_dir")
        mkdir -p "$eval_dir"
        "$python_exec" -u experiment.py serve --checkpoint "$run/checkpoint.pt" --port "$port" > "$eval_dir/server.log" 2>&1 &
        server=$!
        trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
        for attempt in {1..60}; do
            if grep -q "^READY $port " "$eval_dir/server.log"; then break; fi
            if ! kill -0 "$server" 2>/dev/null || [[ $attempt == 60 ]]; then
                cat "$eval_dir/server.log"
                exit 1
            fi
            sleep 1
        done
        cd "$root/ForceHarness"
        "$python_exec" -u scripts/baselines/dp/eval_policy.py --task peg_in_hole_hard \
            --name "act_$mode" --ports "$port" --seeds "$seeds" --workers "$workers" --run "$eval_dir" --egl "${EGL_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
        cd "$root"
        "$python_exec" summarize.py "$eval_dir/results.jsonl" --seeds "$seeds"
        ;;
    compare)
        "$python_exec" summarize.py "$root/runs/paired/current/eval/results.jsonl" \
            "$root/runs/paired/history/eval/results.jsonl" --seeds 9100-9149
        ;;
    *) exit 2 ;;
esac
