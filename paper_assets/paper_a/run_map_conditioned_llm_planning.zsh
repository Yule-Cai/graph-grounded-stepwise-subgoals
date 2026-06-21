#!/usr/bin/env zsh
set -eo pipefail

EPISODES="${1:-20}"
TIMESTEPS="${2:-5000000}"
LM_MODEL="${LM_MODEL:-lfm2.5-8b-a1b-mlx}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/map_conditioned_$(date +%Y%m%d_%H%M%S)}"
SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"

source "$WORKSPACE/paper_a_experiments_desktop/scripts/_paper_a_common.sh"

mkdir -p "$OUT_DIR"
print_header "Paper A map-conditioned LLM waypoint planning"
echo "episodes=$EPISODES timesteps=$TIMESTEPS lm_model=$LM_MODEL lm_url=$LM_URL"
echo "logs=$OUT_DIR"

SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
ALGOS=(${=ALGOS:-ppo sac})

for scenario in "${SCENARIOS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    model_file="models/${algo:u}/${algo}_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip"
    cases="$ROOT/experiments/paper_a/cases/proactive_${scenario}_100.csv"
    log="$OUT_DIR/map_conditioned_${scenario}_${algo}_${LM_MODEL//\//_}_${EPISODES}ep.log"
    echo "RUN scenario=$scenario algo=$algo" | tee -a "$OUT_DIR/_status.log"
    run_py "$SCRIPT" \
      --algo "$algo" \
      --model "$model_file" \
      --cases "$cases" \
      --episodes "$EPISODES" \
      --lm-studio-url "$LM_URL" \
      --lm-model "$LM_MODEL" \
      --repair-invalid-route \
      > "$log" 2>&1
    tail -n 18 "$log"
  done
done

echo "Logs saved to $OUT_DIR"
