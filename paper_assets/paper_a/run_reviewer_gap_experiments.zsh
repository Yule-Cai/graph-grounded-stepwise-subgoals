#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
BASE_OUT_DIR="${BASE_OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/reviewer_gap_${STAMP}}"

RUN_SCORER_BASELINES="${RUN_SCORER_BASELINES:-1}"
RUN_CLOSED_LOOP_PREFERENCE="${RUN_CLOSED_LOOP_PREFERENCE:-1}"
RUN_NEMOTRON_MULTIMAP="${RUN_NEMOTRON_MULTIMAP:-1}"
RUN_RISK_SENSITIVITY="${RUN_RISK_SENSITIVITY:-1}"

mkdir -p "$BASE_OUT_DIR"

echo "Paper A reviewer-gap experiments"
echo "base_out_dir=$BASE_OUT_DIR"
echo "episodes=$EPISODES timesteps=$TIMESTEPS"
echo "stages scorer=$RUN_SCORER_BASELINES closed_loop_pref=$RUN_CLOSED_LOOP_PREFERENCE nemotron_multimap=$RUN_NEMOTRON_MULTIMAP risk=$RUN_RISK_SENSITIVITY"

if [[ "$RUN_SCORER_BASELINES" == "1" ]]; then
  OUT_DIR="$BASE_OUT_DIR/deterministic_scorer_baselines" \
  zsh "$WORKSPACE/paper_assets/paper_a/run_deterministic_scorer_baselines.zsh" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_CLOSED_LOOP_PREFERENCE" == "1" ]]; then
  OUT_DIR="$BASE_OUT_DIR/closed_loop_preference" \
  MODEL_LIST="${MODEL_LIST:-nvidia/nemotron-3-nano-4b google/gemma-4-e4b}" \
  zsh "$WORKSPACE/paper_assets/paper_a/run_closed_loop_preference_experiments.zsh" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_NEMOTRON_MULTIMAP" == "1" ]]; then
  OUT_DIR="${MULTIMAP_OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/multimap_fastpass_20260604}" \
  GENERATE_CASES=0 \
  SKIP_COMPLETED=1 \
  MIN_COMPLETED_ROWS="$EPISODES" \
  MAP_LIST="${MAP_LIST:-reference_villa_ground studio_apartment townhouse_long luxury_villa}" \
  SCENARIOS="${SCENARIOS:-long_horizon semantic_constraint}" \
  SEEDS="${SEEDS:-31 47 73}" \
  TOP_MODELS="nvidia/nemotron-3-nano-4b" \
  RUN_NO_LLM=0 \
  RUN_LLM_STEP=1 \
  RUN_LLM_RAW=0 \
  RUN_ORDER_ENSEMBLE=0 \
  RUN_SHUFFLE_ENSEMBLE=0 \
  zsh "$WORKSPACE/paper_assets/paper_a/run_multimap_generalization_experiments.zsh" "$EPISODES" "$TIMESTEPS"
fi

if [[ "$RUN_RISK_SENSITIVITY" == "1" ]]; then
  OUT_DIR="$BASE_OUT_DIR/risk_weight_sensitivity" \
  zsh "$WORKSPACE/paper_assets/paper_a/run_risk_weight_sensitivity.zsh" "$EPISODES" "$TIMESTEPS"
fi

echo "Reviewer-gap experiments complete: $BASE_OUT_DIR"
