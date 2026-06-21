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
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/graph_perturbation_stress_${STAMP}}"
MAP_LIST=(${=MAP_LIST:-reference_family_flat reference_villa_ground studio_apartment})
SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
MAIN_ALGO="${MAIN_ALGO:-ppo}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
MODEL_LIST=(${=MODEL_LIST:-nvidia/nemotron-3-nano-4b liquid/lfm2.5-1.2b})
BASELINE_MODES=(${=BASELINE_MODES:-no_llm first_candidate preference_scorer})
LLM_MODES=(${=LLM_MODES:-llm_step_order_ensemble llm_step_consistency_gate})
PERTURBATIONS=(${=PERTURBATIONS:-clean edge_drop risk_noise combined})
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_LLM="${RUN_LLM:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-$EPISODES}"
ORDER_GATE_VARIANTS="${ORDER_GATE_VARIANTS:-5}"
ORDER_GATE_MIN_VOTES="${ORDER_GATE_MIN_VOTES:-3}"
ORDER_GATE_MIN_CONSISTENCY="${ORDER_GATE_MIN_CONSISTENCY:-0.60}"
EDGE_DROP_RATE="${EDGE_DROP_RATE:-0.15}"
RISK_CENTER_NOISE="${RISK_CENTER_NOISE:-0.80}"
RISK_RADIUS_SCALE="${RISK_RADIUS_SCALE:-1.15}"

MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs"

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

case_file_for() {
  local map="$1"
  local scenario="$2"
  if [[ "$map" == "reference_family_flat" ]]; then
    if [[ "$scenario" == "long_horizon" ]]; then
      echo "$DESKTOP_ROOT/experiments/paper_a/cases/proactive_long_horizon_100.csv"
    else
      echo "$DESKTOP_ROOT/experiments/paper_a/cases/proactive_semantic_constraint_100.csv"
    fi
  else
    echo "$DESKTOP_ROOT/experiments/paper_a/cases/multimap_generalization/proactive_${map}_${scenario}_100.csv"
  fi
}

perturb_args() {
  local perturb="$1"
  case "$perturb" in
    clean) echo "--graph-edge-drop-rate 0 --risk-center-noise 0 --risk-radius-scale 1.0" ;;
    edge_drop) echo "--graph-edge-drop-rate $EDGE_DROP_RATE --risk-center-noise 0 --risk-radius-scale 1.0" ;;
    risk_noise) echo "--graph-edge-drop-rate 0 --risk-center-noise $RISK_CENTER_NOISE --risk-radius-scale $RISK_RADIUS_SCALE" ;;
    combined) echo "--graph-edge-drop-rate $EDGE_DROP_RATE --risk-center-noise $RISK_CENTER_NOISE --risk-radius-scale $RISK_RADIUS_SCALE" ;;
    *) echo "--graph-edge-drop-rate 0 --risk-center-noise 0 --risk-radius-scale 1.0" ;;
  esac
}

total_jobs=0
if [[ "$RUN_BASELINES" == "1" ]]; then
  total_jobs=$((total_jobs + ${#MAP_LIST[@]} * ${#SCENARIOS[@]} * ${#PERTURBATIONS[@]} * ${#BASELINE_MODES[@]}))
fi
if [[ "$RUN_LLM" == "1" ]]; then
  total_jobs=$((total_jobs + ${#MAP_LIST[@]} * ${#SCENARIOS[@]} * ${#PERTURBATIONS[@]} * ${#LLM_MODES[@]} * ${#MODEL_LIST[@]}))
fi
done_jobs=0
started_at="$(date +%s)"

progress_bar() {
  local done="$1"
  local total="$2"
  local width="${3:-28}"
  local pct=0 filled=0 empty=0
  if (( total > 0 )); then
    pct=$((100 * done / total))
    filled=$((width * done / total))
  fi
  if (( filled > width )); then filled="$width"; fi
  empty=$((width - filled))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3d%%' "$pct"
}

format_duration() {
  local seconds="$1"
  if (( seconds < 0 )); then seconds=0; fi
  printf '%02dh%02dm' "$((seconds / 3600))" "$(((seconds % 3600) / 60))"
}

progress() {
  local msg="$1"
  local now="$(date +%s)"
  local elapsed=$((now - started_at))
  local eta=0
  if (( done_jobs > 0 && total_jobs > done_jobs )); then
    eta=$((elapsed * (total_jobs - done_jobs) / done_jobs))
  fi
  printf '%s %d/%d elapsed=%s eta=%s  %s\n' \
    "$(progress_bar "$done_jobs" "$total_jobs" 30)" \
    "$done_jobs" "$total_jobs" "$(format_duration "$elapsed")" "$(format_duration "$eta")" "$msg"
}

run_condition() {
  local map="$1"
  local scenario="$2"
  local perturb="$3"
  local mode="$4"
  local lm_model="$5"
  local case_file="$(case_file_for "$map" "$scenario")"
  local model_tag="$(sanitize "$lm_model")"
  local model_file="$(model_file_for_algo "$MAIN_ALGO")"
  local label="${map}_${scenario}_${perturb}_${MAIN_ALGO}_${mode}_${model_tag}"
  local episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
  local summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"
  local log="$OUT_DIR/stage_logs/${label}_${EPISODES}ep.log"
  done_jobs=$((done_jobs + 1))
  if [[ ! -f "$case_file" ]]; then
    progress "MISSING cases map=$map scenario=$scenario file=$case_file"
    return 0
  fi
  if [[ "$SKIP_COMPLETED" == "1" ]]; then
    local rows="$(csv_data_rows "$episode_csv")"
    if (( rows >= MIN_COMPLETED_ROWS )); then
      progress "SKIP completed $label rows=$rows"
      return 0
    fi
  fi
  progress "RUN $label"
  local p_arg_string="$(perturb_args "$perturb")"
  local -a p_args
  p_args=(${=p_arg_string})
  PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
    --algo "$MAIN_ALGO" \
    --model "$model_file" \
    --cases "$case_file" \
    --episodes "$EPISODES" \
    --seed 31 \
    --maps "$map" \
    --map-source "gazebo_3d_projection" \
    --lm-studio-url "$LM_URL" \
    --lm-model "$lm_model" \
    --planner-mode "$mode" \
    --order-gate-variants "$ORDER_GATE_VARIANTS" \
    --order-gate-min-votes "$ORDER_GATE_MIN_VOTES" \
    --order-gate-min-consistency "$ORDER_GATE_MIN_CONSISTENCY" \
    "${p_args[@]}" \
    --repair-invalid-route \
    --run-label "$label" \
    --episode-csv "$episode_csv" \
    --summary-csv "$summary_csv" \
    2>&1 | tee "$log"
}

print_header "Paper A graph perturbation stress experiments"
echo "out_dir=$OUT_DIR episodes=$EPISODES timesteps=$TIMESTEPS algo=$MAIN_ALGO"
echo "maps=${MAP_LIST[*]} scenarios=${SCENARIOS[*]}"
echo "perturbations=${PERTURBATIONS[*]} edge_drop=$EDGE_DROP_RATE risk_noise=$RISK_CENTER_NOISE risk_radius_scale=$RISK_RADIUS_SCALE"
echo "baselines=${BASELINE_MODES[*]} llm_modes=${LLM_MODES[*]} models=${MODEL_LIST[*]}"
echo "total_jobs=$total_jobs"

if [[ "$RUN_BASELINES" == "1" ]]; then
  print_header "Stage 1: deterministic baselines"
  for map in "${MAP_LIST[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      for perturb in "${PERTURBATIONS[@]}"; do
        for mode in "${BASELINE_MODES[@]}"; do
          run_condition "$map" "$scenario" "$perturb" "$mode" "no_llm"
        done
      done
    done
  done
fi

if [[ "$RUN_LLM" == "1" ]]; then
  print_header "Stage 2: order-robust LLM planners"
  for map in "${MAP_LIST[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      for perturb in "${PERTURBATIONS[@]}"; do
        for model in "${MODEL_LIST[@]}"; do
          for mode in "${LLM_MODES[@]}"; do
            run_condition "$map" "$scenario" "$perturb" "$mode" "$model"
          done
        done
      done
    done
  done
fi

print_header "Stage 3: summarize"
run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
  --out-csv "$OUT_DIR/graph_perturbation_summary_with_ci.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

echo "Graph perturbation stress experiments complete: $OUT_DIR"
