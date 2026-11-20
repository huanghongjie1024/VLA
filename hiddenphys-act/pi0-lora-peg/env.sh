export FH_ROOT="${FH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export OPENPI_ROOT="${OPENPI_ROOT:-$FH_ROOT/../openpi}"
export PI_PY="${PI_PY:-$OPENPI_ROOT/.venv/bin/python}"
export FORCEHARNESS="$FH_ROOT/ForceHarness"
export PI_WORK="$FH_ROOT/pi0-lora-peg"
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
export PI0_PARAMS="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params"
export HF_LEROBOT_HOME="$PI_WORK/lerobot-data"
export HF_DATASETS_CACHE="$PI_WORK/arrow-cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export FH_REPO_ID=local/hiddenphys_peg_pi0
export FH_PI05_WORK="$PI_WORK"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PI_SIM_PY="${PI_SIM_PY:-${HIDDENPHYS_PYTHON:-python}}"
