#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/supplemental_reviewer_${STAMP}}"
RUNNER="$WORKSPACE/paper_assets/paper_a/run_final_aaai_experiments.zsh"

TOP_MODELS="${TOP_MODELS:-liquid/lfm2.5-1.2b nvidia/nemotron-3-nano-4b}"
ALL_SEEDS=(${=SEEDS:-31})
SCENARIOS="${SCENARIOS:-long_horizon semantic_constraint}"
ABLATIONS=(${=ABLATIONS:-no_risk no_hop no_progress shuffle_order})
RUN_HEURISTIC_BASELINES="${RUN_HEURISTIC_BASELINES:-1}"

mkdir -p "$OUT_DIR"
echo "Supplemental reviewer experiments"
echo "out_dir=$OUT_DIR"
echo "episodes=$EPISODES timesteps=$TIMESTEPS"
echo "scenarios=$SCENARIOS"
echo "top_models=$TOP_MODELS"
echo "seeds=${ALL_SEEDS[*]}"
echo "run_heuristic_baselines=$RUN_HEURISTIC_BASELINES"
echo "ablations=${ABLATIONS[*]}"

run_stage() {
  local name="$1"
  shift
  echo "============================================================"
  echo "$name"
  echo "============================================================"
  "$@"
}

if [[ "$RUN_HEURISTIC_BASELINES" == "1" ]]; then
  for seed in "${ALL_SEEDS[@]}"; do
    run_stage "Stage A: non-LLM legal-neighbor baselines seed=$seed" env \
      OUT_DIR="$OUT_DIR" \
      SEED="$seed" \
      SKIP_COMPLETED=1 \
      MIN_COMPLETED_ROWS="$EPISODES" \
      INCLUDE_HF_EXTERNAL_MODELS=0 \
      RUN_NO_LLM=0 \
      RUN_MODEL_SWEEP=1 \
      RUN_CONTROLLER_ABLATION=0 \
      RUN_PACKAGE=0 \
      MODEL_LIST="no_llm" \
      PRIMARY_MODEL="no_llm" \
      MAIN_ALGO="ppo" \
      PLANNER_MODES="graph_shortest greedy_progress greedy_hop greedy_risk random_legal" \
      SCENARIOS="$SCENARIOS" \
      zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
  done
fi

if [[ "${RUN_LLM_RETRY:-1}" == "1" ]]; then
  for seed in "${ALL_SEEDS[@]}"; do
    run_stage "Stage B: LLM step retry for top models seed=$seed" env \
      OUT_DIR="$OUT_DIR" \
      SEED="$seed" \
      SKIP_COMPLETED=1 \
      MIN_COMPLETED_ROWS="$EPISODES" \
      INCLUDE_HF_EXTERNAL_MODELS=0 \
      RUN_NO_LLM=0 \
      RUN_MODEL_SWEEP=1 \
      RUN_CONTROLLER_ABLATION=0 \
      RUN_PACKAGE=0 \
      MODEL_LIST="$TOP_MODELS" \
      PRIMARY_MODEL="${TOP_MODELS%% *}" \
      MAIN_ALGO="ppo" \
      PLANNER_MODES="llm_step_retry" \
      LLM_RETRIES="${LLM_RETRIES:-2}" \
      SCENARIOS="$SCENARIOS" \
      zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
  done
fi

if [[ "${RUN_FEATURE_ABLATION:-0}" == "1" ]]; then
  for ablation in "${ABLATIONS[@]}"; do
    for seed in "${ALL_SEEDS[@]}"; do
      run_stage "Stage C: feature ablation=$ablation seed=$seed" env \
        OUT_DIR="$OUT_DIR" \
        SEED="$seed" \
        SKIP_COMPLETED=1 \
        MIN_COMPLETED_ROWS="$EPISODES" \
        INCLUDE_HF_EXTERNAL_MODELS=0 \
        RUN_NO_LLM=0 \
        RUN_MODEL_SWEEP=1 \
        RUN_CONTROLLER_ABLATION=0 \
        RUN_PACKAGE=0 \
        MODEL_LIST="$TOP_MODELS" \
        PRIMARY_MODEL="${TOP_MODELS%% *}" \
        MAIN_ALGO="ppo" \
        PLANNER_MODES="llm_step" \
        CANDIDATE_FEATURE_ABLATION="$ablation" \
        SCENARIOS="$SCENARIOS" \
        zsh "$RUNNER" "$EPISODES" "$TIMESTEPS"
    done
  done
fi

echo "Supplemental experiments complete: $OUT_DIR"
