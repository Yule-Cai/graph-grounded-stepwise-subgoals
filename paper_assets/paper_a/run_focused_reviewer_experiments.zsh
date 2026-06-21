#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$WORKSPACE/paper_assets/paper_a/run_final_aaai_experiments.zsh"

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/focused_reviewer_${STAMP}}"
SCENARIOS="${SCENARIOS:-long_horizon semantic_constraint}"
SEED="${SEED:-31}"
FOCUSED_MODEL="${FOCUSED_MODEL:-liquid/lfm2.5-1.2b}"
RUN_REFERENCE_NO_LLM="${RUN_REFERENCE_NO_LLM:-1}"
RUN_RAW_VS_STEP="${RUN_RAW_VS_STEP:-1}"
RUN_SHUFFLE_STEP="${RUN_SHUFFLE_STEP:-1}"
RUN_ORDER_ENSEMBLE="${RUN_ORDER_ENSEMBLE:-1}"
RUN_CONSISTENCY_GATE="${RUN_CONSISTENCY_GATE:-1}"
RUN_SHUFFLE_ENSEMBLE="${RUN_SHUFFLE_ENSEMBLE:-0}"
ORDER_GATE_VARIANTS="${ORDER_GATE_VARIANTS:-3}"
ORDER_GATE_MIN_VOTES="${ORDER_GATE_MIN_VOTES:-2}"
ORDER_GATE_MIN_CONSISTENCY="${ORDER_GATE_MIN_CONSISTENCY:-0.67}"

mkdir -p "$OUT_DIR"

run_stage() {
  local name="$1"
  shift
  echo "============================================================"
  echo "$name"
  echo "============================================================"
  "$@"
}

common_env=(
  OUT_DIR="$OUT_DIR"
  SEED="$SEED"
  SKIP_COMPLETED=1
  MIN_COMPLETED_ROWS="$EPISODES"
  INCLUDE_HF_EXTERNAL_MODELS=0
  RUN_CONTROLLER_ABLATION=0
  RUN_PACKAGE=0
  MAIN_ALGO=ppo
  SCENARIOS="$SCENARIOS"
)

echo "Focused Paper A reviewer experiments"
echo "out_dir=$OUT_DIR"
echo "episodes=$EPISODES timesteps=$TIMESTEPS seed=$SEED"
echo "scenarios=$SCENARIOS model=$FOCUSED_MODEL"
echo "order_gate_variants=$ORDER_GATE_VARIANTS min_votes=$ORDER_GATE_MIN_VOTES min_consistency=$ORDER_GATE_MIN_CONSISTENCY"

if [[ "$RUN_REFERENCE_NO_LLM" == "1" ]]; then
  run_stage "Stage A: deterministic reference baseline" env \
    "${common_env[@]}" \
    RUN_NO_LLM=1 \
    RUN_MODEL_SWEEP=0 \
    NO_LLM_ALGOS=ppo \
    PRIMARY_MODEL="$FOCUSED_MODEL" \
    MODEL_LIST="$FOCUSED_MODEL" \
    CANDIDATE_FEATURE_ABLATION=none \
    zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_RAW_VS_STEP" == "1" ]]; then
  run_stage "Stage B: raw full-route LLM vs stepwise LLM" env \
    "${common_env[@]}" \
    RUN_NO_LLM=0 \
    RUN_MODEL_SWEEP=1 \
    PRIMARY_MODEL="$FOCUSED_MODEL" \
    MODEL_LIST="$FOCUSED_MODEL" \
    PLANNER_MODES="llm_raw llm_step" \
    CANDIDATE_FEATURE_ABLATION=none \
    zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_SHUFFLE_STEP" == "1" ]]; then
  run_stage "Stage C: candidate-order sensitivity for stepwise LLM" env \
    "${common_env[@]}" \
    RUN_NO_LLM=0 \
    RUN_MODEL_SWEEP=1 \
    PRIMARY_MODEL="$FOCUSED_MODEL" \
    MODEL_LIST="$FOCUSED_MODEL" \
    PLANNER_MODES="llm_step" \
    CANDIDATE_FEATURE_ABLATION=shuffle_order \
    zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_ORDER_ENSEMBLE" == "1" ]]; then
  run_stage "Stage D: order-ensemble mitigation" env \
    "${common_env[@]}" \
    RUN_NO_LLM=0 \
    RUN_MODEL_SWEEP=1 \
    PRIMARY_MODEL="$FOCUSED_MODEL" \
    MODEL_LIST="$FOCUSED_MODEL" \
    PLANNER_MODES="llm_step_order_ensemble" \
    CANDIDATE_FEATURE_ABLATION=none \
    zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_CONSISTENCY_GATE" == "1" ]]; then
  run_stage "Stage E: order-consistency gated fallback" env \
    "${common_env[@]}" \
    RUN_NO_LLM=0 \
    RUN_MODEL_SWEEP=1 \
    PRIMARY_MODEL="$FOCUSED_MODEL" \
    MODEL_LIST="$FOCUSED_MODEL" \
    PLANNER_MODES="llm_step_consistency_gate" \
    ORDER_GATE_VARIANTS="$ORDER_GATE_VARIANTS" \
    ORDER_GATE_MIN_VOTES="$ORDER_GATE_MIN_VOTES" \
    ORDER_GATE_MIN_CONSISTENCY="$ORDER_GATE_MIN_CONSISTENCY" \
    CANDIDATE_FEATURE_ABLATION=none \
    zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_SHUFFLE_ENSEMBLE" == "1" ]]; then
  run_stage "Stage F: shuffled order-ensemble mitigation" env \
    "${common_env[@]}" \
    RUN_NO_LLM=0 \
    RUN_MODEL_SWEEP=1 \
    PRIMARY_MODEL="$FOCUSED_MODEL" \
    MODEL_LIST="$FOCUSED_MODEL" \
    PLANNER_MODES="llm_step_order_ensemble" \
    CANDIDATE_FEATURE_ABLATION=shuffle_order \
    zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
fi

echo "Focused experiments complete: $OUT_DIR"
