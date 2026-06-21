#!/usr/bin/env zsh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: set OPENROUTER_API_KEY first."
  echo "Set the OpenRouter API key in the environment before running."
  exit 2
fi

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/paper_assets/paper_a/rerun_logs/openrouter_gptoss120b_${STAMP}}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-openai/gpt-oss-120b:free}"
LM_URL="${LM_URL:-https://openrouter.ai/api/v1}"

RUN_STRESS="${RUN_STRESS:-1}"
RUN_CLOSED_LOOP="${RUN_CLOSED_LOOP:-0}"
TIMESTEPS="${TIMESTEPS:-5000000}"

echo "Paper A OpenRouter GPT-OSS-120B sanity check"
echo "out_root=$OUT_ROOT"
echo "model=$OPENROUTER_MODEL"
echo "lm_url=$LM_URL"
echo "run_stress=$RUN_STRESS run_closed_loop=$RUN_CLOSED_LOOP"

if [[ "$RUN_STRESS" == "1" ]]; then
  OUT_DIR="$OUT_ROOT/language_preference_stress" \
  LM_URL="$LM_URL" \
  MODEL_LIST="$OPENROUTER_MODEL" \
  BASE_LIMIT="${STRESS_BASE_LIMIT:-20}" \
  VARIANTS_PER_TASK="${STRESS_VARIANTS_PER_TASK:-2}" \
  BENCHMARK="${STRESS_BENCHMARK:-all}" \
  OPTION_ORDER="${STRESS_OPTION_ORDER:-shuffled}" \
  TIMEOUT_S="${STRESS_TIMEOUT_S:-90}" \
  MAX_TOKENS="${STRESS_MAX_TOKENS:-256}" \
  zsh "$WORKSPACE/paper_assets/paper_a/run_language_preference_stress_experiments.zsh"
fi

if [[ "$RUN_CLOSED_LOOP" == "1" ]]; then
  OUT_DIR="$OUT_ROOT/preference_250_closed_loop" \
  CASE_INPUT="${CASE_INPUT:-$WORKSPACE/paper_a_experiments_desktop/experiments/paper_a/cases/proactive_semantic_constraint_100.csv}" \
  LM_URL="$LM_URL" \
  MODEL_LIST="$OPENROUTER_MODEL" \
  RUN_DETERMINISTIC_BASELINES="${RUN_DETERMINISTIC_BASELINES:-0}" \
  RUN_LLM_BASELINES=1 \
  BASE_LIMIT="${CLOSED_LOOP_BASE_LIMIT:-50}" \
  PREFERENCE_SET="${CLOSED_LOOP_PREFERENCE_SET:-basic}" \
  BASELINE_MODES="${BASELINE_MODES:-no_llm graph_shortest}" \
  LLM_BASELINE_MODES="${LLM_BASELINE_MODES:-language_to_cost route_option_rank}" \
  SKIP_COMPLETED="${SKIP_COMPLETED:-1}" \
  MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-${CLOSED_LOOP_EPISODES:-250}}" \
  zsh "$WORKSPACE/paper_assets/paper_a/run_preference_250_upgrade.zsh" "${CLOSED_LOOP_EPISODES:-250}" "$TIMESTEPS"
fi

echo "OpenRouter sanity check complete: $OUT_ROOT"
