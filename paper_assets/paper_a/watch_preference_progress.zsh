#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/preference_250_extended_models_20260620}"
EPISODES="${EPISODES:-250}"
MODEL_LIST="${MODEL_LIST:-google/gemma-4-12b google/gemma-4-e4b google/gemma-3-1b qwen/qwen3-1.7b qwenpaw-flash-9b}"
LLM_BASELINE_MODES="${LLM_BASELINE_MODES:-language_to_cost route_option_rank}"
REFRESH_SECONDS="${REFRESH_SECONDS:-5}"

exec python3 "$WORKSPACE/paper_assets/paper_a/watch_preference_progress.py" \
  --out-dir "$OUT_DIR" \
  --episodes-per-job "$EPISODES" \
  --models "$MODEL_LIST" \
  --modes "$LLM_BASELINE_MODES" \
  --refresh "$REFRESH_SECONDS" \
  "$@"
