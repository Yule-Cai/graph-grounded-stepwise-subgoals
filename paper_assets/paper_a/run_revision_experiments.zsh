#!/usr/bin/env zsh
set -eo pipefail

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
LM_MODEL="${LM_MODEL:-lfm2.5-8b-a1b-mlx}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/revision_$(date +%Y%m%d_%H%M%S)}"
SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"

source "$WORKSPACE/paper_a_experiments_desktop/scripts/_paper_a_common.sh"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries"
print_header "Paper A revision experiments"
echo "episodes=$EPISODES timesteps=$TIMESTEPS lm_model=$LM_MODEL lm_url=$LM_URL"
echo "logs=$OUT_DIR"

SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
ALGOS=(${=ALGOS:-ppo})
PLANNER_MODES=(${=PLANNER_MODES:-no_llm llm_retry})
LLM_RETRIES="${LLM_RETRIES:-2}"

for scenario in "${SCENARIOS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    model_file="models/${algo:u}/${algo}_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip"
    cases="$ROOT/experiments/paper_a/cases/proactive_${scenario}_100.csv"
    for mode in "${PLANNER_MODES[@]}"; do
      label="${scenario}_${algo}_${mode}"
      log="$OUT_DIR/${label}_${EPISODES}ep.log"
      episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
      summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"
      echo "RUN scenario=$scenario algo=$algo mode=$mode" | tee -a "$OUT_DIR/_status.log"
      run_py "$SCRIPT" \
        --algo "$algo" \
        --model "$model_file" \
        --cases "$cases" \
        --episodes "$EPISODES" \
        --lm-studio-url "$LM_URL" \
        --lm-model "$LM_MODEL" \
        --planner-mode "$mode" \
        --llm-retries "$LLM_RETRIES" \
        --repair-invalid-route \
        --run-label "$label" \
        --episode-csv "$episode_csv" \
        --summary-csv "$summary_csv" \
        > "$log" 2>&1
      tail -n 24 "$log"
    done
  done
done

run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --out-csv "$OUT_DIR/revision_experiment_summary_with_ci.csv"

echo "Logs saved to $OUT_DIR"
echo "Combined summary: $OUT_DIR/revision_experiment_summary_with_ci.csv"
