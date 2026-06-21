#!/usr/bin/env zsh
set -uo pipefail

WORKSPACE="${WORKSPACE:-$(pwd)}"
DESKTOP_ROOT="${DESKTOP_ROOT:-$WORKSPACE/paper_a_experiments_desktop}"
source "$DESKTOP_ROOT/scripts/_paper_a_common.sh"

EPISODES="${1:-100}"
STEPS="${2:-5000000}"
LM_MODEL="${LM_MODEL:-lfm2.5-8b-a1b-mlx}"
LM_STUDIO_URL="${LM_STUDIO_URL:-http://127.0.0.1:1234}"
SCENARIOS=(${=PAPER_A_SCENARIOS:-long_horizon semantic_constraint})
ALGOS=(${=PAPER_A_ALGOS:-ppo sac})
STAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${LOG_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/lfm25_a1b_mlx_${STAMP}}"
mkdir -p "$LOG_DIR"

sanitize() {
  echo "$1" | sed 's#[/: ]#_#g' | sed 's#[^A-Za-z0-9_.-]#_#g'
}

model_file_for_algo() {
  local algo="$1"
  case "$algo" in
    ppo) echo "models/PPO/ppo_reference_family_flat_goalfirst_${STEPS}_from_scratch.zip" ;;
    sac) echo "models/SAC/sac_reference_family_flat_goalfirst_${STEPS}_from_scratch.zip" ;;
    *) echo "ERROR_UNKNOWN_ALGO" ;;
  esac
}

run_one() {
  local scenario="$1"
  local algo="$2"
  local case_file="$ROOT/experiments/paper_a/cases/proactive_${scenario}_${EPISODES}.csv"
  local model_file
  model_file="$(model_file_for_algo "$algo")"
  local tag
  tag="$(sanitize "lmstudio_${scenario}_${algo}_${LM_MODEL}_${EPISODES}ep")"
  local log_file="$LOG_DIR/${tag}.log"

  echo "============================================================"
  echo "START $tag"
  echo "LOG=$log_file"
  echo "LM_STUDIO_URL=$LM_STUDIO_URL"
  echo "LM_MODEL=$LM_MODEL"
  echo "============================================================"

  run_py -m llm_rl_nav.training.eval_paper_a_proactive_route \
    --mode llm_route \
    --algo "$algo" \
    --model "$model_file" \
    --cases "$case_file" \
    --episodes "$EPISODES" \
    --map-source gazebo_3d_projection \
    --max-steps 1000 \
    --subgoal-radius 0.85 \
    --reward-profile v8_goal \
    --goal-min-distance 2.0 \
    --goal-max-distance 20.0 \
    --goal-point-probability 0.95 \
    --safety-shield \
    --shield-min-clearance 0.18 \
    --shield-intervention-penalty 0.25 \
    --waypoint-spacing 2.2 \
    --waypoint-grid-resolution 0.55 \
    --safe-success-threshold 0.75 \
    --lm-studio-url "$LM_STUDIO_URL" \
    --lm-model "$LM_MODEL" \
    --llm-timeout-s 180 \
    --llm-max-tokens 750 \
    --repair-invalid-route \
    --execution-aware-rerank \
    --fast-safe-clearance 0.75 \
    --turn-weight 1.8 \
    --clearance-weight 7.0 \
    --waypoint-count-weight 0.35 \
    --semantic-cost-weight 5.0 2>&1 | tee "$log_file"

  local exit_code="${pipestatus[1]}"
  if [[ "$exit_code" -eq 0 ]]; then
    echo "OK: $tag" | tee -a "$LOG_DIR/_status.log"
  else
    echo "FAIL: $tag code=$exit_code" | tee -a "$LOG_DIR/_status.log"
  fi
}

echo "model list at start:" > "$LOG_DIR/lmstudio_models_at_start.json"
curl -s "$LM_STUDIO_URL/v1/models" >> "$LOG_DIR/lmstudio_models_at_start.json" || true

for scenario in "${SCENARIOS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    run_one "$scenario" "$algo"
  done
done

"$ROOT/scripts/summarize_paper_a_proactive_logs.py" "$LOG_DIR" | tee "$LOG_DIR/paper_a_proactive_summary.txt" || true
echo "Logs saved to $LOG_DIR"
