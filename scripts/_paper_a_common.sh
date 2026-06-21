#!/usr/bin/env zsh
# Shared helpers for Paper A desktop experiments. Source this file from zsh scripts.
set -eo pipefail

_COMMON_FILE="${(%):-%x}"
SCRIPT_DIR="$(cd "$(dirname "$_COMMON_FILE")" && pwd)"
export ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export CONDA_ENV="${CONDA_ENV:-ros_env}"

cd "$ROOT"
export LLM_RL_NAV_HOME="$ROOT"
export ROS_LOG_DIR="$ROOT/log/ros"
export MPLCONFIGDIR="$ROOT/log/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/src/llm_rl_nav:$ROOT/src:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs/eval" "$ROOT/log/ros" "$ROOT/log/matplotlib" "$ROOT/experiments/paper_a/results" "$ROOT/experiments/paper_a/cases"

ensure_conda_available() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]]; then
    source "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
}

run_py() {
  # If ros_env is already active, run directly. Otherwise run through conda.
  if [[ "${PAPER_A_NO_CONDA:-0}" == "1" || "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    env PYTHONPATH="$PYTHONPATH" LLM_RL_NAV_HOME="$LLM_RL_NAV_HOME" MPLCONFIGDIR="$MPLCONFIGDIR" python "$@"
  else
    ensure_conda_available
    if ! command -v conda >/dev/null 2>&1; then
      echo "ERROR: conda not found. Run: conda activate $CONDA_ENV  OR set PAPER_A_NO_CONDA=1" >&2
      exit 2
    fi
    conda run --no-capture-output -n "$CONDA_ENV" env \
      PYTHONPATH="$PYTHONPATH" \
      LLM_RL_NAV_HOME="$LLM_RL_NAV_HOME" \
      MPLCONFIGDIR="$MPLCONFIGDIR" \
      TOKENIZERS_PARALLELISM=false \
      python "$@"
  fi
}

print_header() {
  echo "============================================================"
  echo "$1"
  echo "ROOT=$ROOT"
  echo "CONDA_ENV=$CONDA_ENV"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "============================================================"
}
