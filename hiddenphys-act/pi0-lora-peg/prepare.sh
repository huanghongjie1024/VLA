#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
mkdir -p "$PI_WORK/raw/peg_in_hole_hard"
shopt -s nullglob
demos=("$FH_ROOT"/demos/episode_*.npz)
test "${#demos[@]}" -eq 185
for f in "${demos[@]}"; do
    name=${f##*/}
    seed=${name#episode_}
    ln -sfn "$f" "$PI_WORK/raw/peg_in_hole_hard/s${seed%.npz}.npz"
done

if [[ ! -f "$HF_LEROBOT_HOME/$FH_REPO_ID/meta_fh.json" ]]; then
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PI_PY"         "$FORCEHARNESS/scripts/baselines/pi05/convert_to_lerobot.py"         --demos "$PI_WORK/raw" --tasks peg_in_hole_hard --scene-pad 15 --repo-id "$FH_REPO_ID"
fi
"$PI_PY" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["HF_LEROBOT_HOME"]) / os.environ["FH_REPO_ID"]
meta = json.loads((root / "meta_fh.json").read_text())
info = json.loads((root / "meta/info.json").read_text())
peg = meta["tasks"]["peg_in_hole_hard"]
assert meta["state_dim"] == 26 and meta["wrench_dim"] == 6
assert info["fps"] == 20 and info["features"]["actions"]["shape"] == [13]
assert peg["n_episodes"] == 185 and peg["n_frames"] == 122194
print("Validated 185 episodes / 122,194 frames / state 26 / wrench 6 / action 13")
PY
