#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/paper_assets/paper_a/rerun_logs/language_value_upgrade_${STAMP}}"
EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"

MODEL_LIST="${MODEL_LIST:-nvidia/nemotron-3-nano-4b google/gemma-4-e4b liquid/lfm2.5-1.2b}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
RUN_STRESS_DIAGNOSTIC="${RUN_STRESS_DIAGNOSTIC:-1}"
RUN_CLOSED_LOOP="${RUN_CLOSED_LOOP:-1}"
BASE_LIMIT_STRESS="${BASE_LIMIT_STRESS:-20}"
VARIANTS_PER_TASK="${VARIANTS_PER_TASK:-2}"
BASE_LIMIT_CLOSED_LOOP="${BASE_LIMIT_CLOSED_LOOP:-10}"
PREFERENCE_SET="${PREFERENCE_SET:-all}"
LLM_MODES="${LLM_MODES:-llm_step_order_ensemble llm_step_consistency_gate}"
BASELINE_MODES="${BASELINE_MODES:-no_llm graph_shortest first_candidate weighted_scorer preference_scorer}"
ORDER_GATE_VARIANTS="${ORDER_GATE_VARIANTS:-5}"
ORDER_GATE_MIN_VOTES="${ORDER_GATE_MIN_VOTES:-3}"
ORDER_GATE_MIN_CONSISTENCY="${ORDER_GATE_MIN_CONSISTENCY:-0.60}"

mkdir -p "$OUT_ROOT"
echo "============================================================"
echo "Paper A language-value upgrade experiments"
echo "out_root=$OUT_ROOT"
echo "episodes=$EPISODES timesteps=$TIMESTEPS"
echo "models=$MODEL_LIST"
echo "run_stress_diagnostic=$RUN_STRESS_DIAGNOSTIC run_closed_loop=$RUN_CLOSED_LOOP"
echo "============================================================"

if [[ "$RUN_STRESS_DIAGNOSTIC" == "1" ]]; then
  OUT_DIR="$OUT_ROOT/language_preference_stress" \
  MODEL_LIST="$MODEL_LIST" \
  LM_URL="$LM_URL" \
  BASE_LIMIT="$BASE_LIMIT_STRESS" \
  VARIANTS_PER_TASK="$VARIANTS_PER_TASK" \
  BENCHMARK="all" \
  OPTION_ORDER="shuffled" \
  zsh "$WORKSPACE/paper_assets/paper_a/run_language_preference_stress_experiments.zsh"
fi

if [[ "$RUN_CLOSED_LOOP" == "1" ]]; then
  OUT_DIR="$OUT_ROOT/closed_loop_preference_order_robust" \
  MODEL_LIST="$MODEL_LIST" \
  LM_URL="$LM_URL" \
  BASE_LIMIT="$BASE_LIMIT_CLOSED_LOOP" \
  PREFERENCE_SET="$PREFERENCE_SET" \
  BASELINE_MODES="$BASELINE_MODES" \
  LLM_MODES="$LLM_MODES" \
  ORDER_GATE_VARIANTS="$ORDER_GATE_VARIANTS" \
  ORDER_GATE_MIN_VOTES="$ORDER_GATE_MIN_VOTES" \
  ORDER_GATE_MIN_CONSISTENCY="$ORDER_GATE_MIN_CONSISTENCY" \
  RUN_BASELINES=1 \
  RUN_LLM_STEP=1 \
  zsh "$WORKSPACE/paper_assets/paper_a/run_closed_loop_preference_experiments.zsh" "$EPISODES" "$TIMESTEPS"
fi

echo "============================================================"
echo "Language-value upgrade experiments complete"
echo "out_root=$OUT_ROOT"
echo "============================================================"
