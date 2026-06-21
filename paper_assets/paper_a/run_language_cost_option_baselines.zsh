#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -z "${DESKTOP_ROOT:-}" ]]; then
  if [[ -f "$WORKSPACE/paper_a_experiments_desktop/scripts/_paper_a_common.sh" ]]; then
    DESKTOP_ROOT="$WORKSPACE/paper_a_experiments_desktop"
  else
    DESKTOP_ROOT="${DESKTOP_ROOT:-$WORKSPACE/paper_a_experiments_desktop}"
  fi
fi
source "$DESKTOP_ROOT/scripts/_paper_a_common.sh"

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/language_cost_option_baselines_${STAMP}}"
BASE_LIMIT="${BASE_LIMIT:-20}"
CASE_INPUT="${CASE_INPUT:-$WORKSPACE/paper_a_experiments_desktop/experiments/paper_a/cases/proactive_semantic_constraint_100.csv}"
MAP_ID="${MAP_ID:-reference_family_flat}"
MAP_SOURCE="${MAP_SOURCE:-gazebo_3d_projection}"
MAIN_ALGO="${MAIN_ALGO:-ppo}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-60}"
MODEL_LIST=(${=MODEL_LIST:-liquid/lfm2.5-1.2b nvidia/nemotron-3-nano-4b})
BASELINE_MODES=(${=BASELINE_MODES:-no_llm graph_shortest weighted_scorer preference_scorer})
LLM_BASELINE_MODES=(${=LLM_BASELINE_MODES:-language_to_cost route_option_rank})
RUN_DETERMINISTIC_BASELINES="${RUN_DETERMINISTIC_BASELINES:-1}"
RUN_LLM_BASELINES="${RUN_LLM_BASELINES:-1}"
PREFERENCE_SET="${PREFERENCE_SET:-stress}"
ROUTE_OPTION_RISK_WEIGHTS="${ROUTE_OPTION_RISK_WEIGHTS:-0,1,2.5,5,10,15}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-$EPISODES}"

MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
CASE_SCRIPT="$WORKSPACE/paper_assets/paper_a/generate_closed_loop_preference_cases.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"
PREFERENCE_SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_preference_by_type.py"
CASE_FILE="$OUT_DIR/cases/language_preference_${BASE_LIMIT}base_${EPISODES}ep.csv"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs" "$OUT_DIR/cases"

model_file_for_algo() {
  local algo="$1"
  case "$algo" in
    ppo) echo "models/PPO/ppo_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    sac) echo "models/SAC/sac_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    *) echo "ERROR_UNKNOWN_ALGO" ;;
  esac
}

sanitize() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

csv_data_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local lines=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do lines=$((lines + 1)); done < "$path"
  if (( lines <= 1 )); then echo 0; else echo $((lines - 1)); fi
}

progress_bar() {
  local done="$1"
  local total="$2"
  local label="$3"
  local width=32
  local percent=0
  if (( total > 0 )); then percent=$((done * 100 / total)); fi
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local bar=""
  local i
  for ((i=0; i<filled; i++)); do bar+="#"; done
  for ((i=0; i<empty; i++)); do bar+="-"; done
  printf '[%s] %3d%% %s (%d/%d)\n' "$bar" "$percent" "$label" "$done" "$total"
}

run_condition() {
  local mode="$1"
  local lm_model="$2"
  local model_tag="$(sanitize "$lm_model")"
  local model_file="$(model_file_for_algo "$MAIN_ALGO")"
  local label="language_cost_option_${MAIN_ALGO}_${mode}_${model_tag}"
  local episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
  local summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"
  local log="$OUT_DIR/${label}_${EPISODES}ep.log"
  if [[ "$SKIP_COMPLETED" == "1" ]]; then
    local rows="$(csv_data_rows "$episode_csv")"
    if (( rows >= MIN_COMPLETED_ROWS )); then
      echo "SKIP completed mode=$mode model=$lm_model rows=$rows"
      return 0
    fi
  fi
  echo "RUN mode=$mode model=$lm_model cases=$CASE_FILE"
  PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
    --algo "$MAIN_ALGO" \
    --model "$model_file" \
    --cases "$CASE_FILE" \
    --episodes "$EPISODES" \
    --seed 31 \
    --maps "$MAP_ID" \
    --map-source "$MAP_SOURCE" \
    --lm-studio-url "$LM_URL" \
    --lm-model "$lm_model" \
    --llm-timeout-s "$LLM_TIMEOUT_S" \
    --planner-mode "$mode" \
    --route-option-risk-weights "$ROUTE_OPTION_RISK_WEIGHTS" \
    --repair-invalid-route \
    --run-label "$label" \
    --episode-csv "$episode_csv" \
    --summary-csv "$summary_csv" \
    2>&1 | tee "$log"
}

print_header "Paper A language-to-cost and route-option LLM baselines"
echo "out_dir=$OUT_DIR"
echo "episodes=$EPISODES base_limit=$BASE_LIMIT preference_set=$PREFERENCE_SET map=$MAP_ID"
echo "case_input=$CASE_INPUT"
echo "llm_timeout_s=$LLM_TIMEOUT_S"
echo "models=${MODEL_LIST[*]}"
echo "deterministic_baselines=${BASELINE_MODES[*]}"
echo "llm_baselines=${LLM_BASELINE_MODES[*]} route_option_risk_weights=$ROUTE_OPTION_RISK_WEIGHTS"

run_py "$CASE_SCRIPT" \
  --input "$CASE_INPUT" \
  --output "$CASE_FILE" \
  --base-limit "$BASE_LIMIT" \
  --map-id "$MAP_ID" \
  --preference-set "$PREFERENCE_SET" \
  2>&1 | tee "$OUT_DIR/stage_logs/generate_cases.log"

total_jobs=0
if [[ "$RUN_DETERMINISTIC_BASELINES" == "1" ]]; then
  total_jobs=$((total_jobs + ${#BASELINE_MODES[@]}))
fi
if [[ "$RUN_LLM_BASELINES" == "1" ]]; then
  total_jobs=$((total_jobs + ${#LLM_BASELINE_MODES[@]} * ${#MODEL_LIST[@]}))
fi
done_jobs=0

if [[ "$RUN_DETERMINISTIC_BASELINES" == "1" ]]; then
  print_header "Stage 1: deterministic baselines"
  for mode in "${BASELINE_MODES[@]}"; do
    progress_bar "$done_jobs" "$total_jobs" "starting deterministic/$mode"
    run_condition "$mode" "no_llm"
    done_jobs=$((done_jobs + 1))
    progress_bar "$done_jobs" "$total_jobs" "finished deterministic/$mode"
  done
fi

if [[ "$RUN_LLM_BASELINES" == "1" ]]; then
  print_header "Stage 2: LLM baselines"
  for model in "${MODEL_LIST[@]}"; do
    for mode in "${LLM_BASELINE_MODES[@]}"; do
      progress_bar "$done_jobs" "$total_jobs" "starting $mode/$model"
      run_condition "$mode" "$model"
      done_jobs=$((done_jobs + 1))
      progress_bar "$done_jobs" "$total_jobs" "finished $mode/$model"
    done
  done
fi

print_header "Stage 3: summarize"
run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
  --out-csv "$OUT_DIR/language_cost_option_summary_with_ci.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

run_py "$PREFERENCE_SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --out-csv "$OUT_DIR/preference_by_type_summary.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/preference_by_type_summary.log"

echo "Language/cost route-option baseline experiments complete: $OUT_DIR"
