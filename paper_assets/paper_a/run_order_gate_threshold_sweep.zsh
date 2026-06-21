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
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/order_gate_threshold_sweep_${STAMP}}"

MAP_LIST=(${=MAP_LIST:-reference_family_flat})
SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
MODEL_LIST=(${=MODEL_LIST:-liquid/lfm2.5-1.2b})
MAIN_ALGO="${MAIN_ALGO:-ppo}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
SEED="${SEED:-31}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-$EPISODES}"

RUN_REFERENCE_ABLATION="${RUN_REFERENCE_ABLATION:-1}"
RUN_GATE_SWEEP="${RUN_GATE_SWEEP:-1}"
GATE_CONFIGS=(${=GATE_CONFIGS:-lenient:3:2:0.50 balanced:5:3:0.60 strict:5:4:0.80})

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
  while IFS= read -r line || [[ -n "$line" ]]; do
    lines=$((lines + 1))
  done < "$path"
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

total_jobs=0
if [[ "$RUN_REFERENCE_ABLATION" == "1" ]]; then
  total_jobs=$((total_jobs + ${#MAP_LIST[@]} * ${#SCENARIOS[@]} * (1 + 3 * ${#MODEL_LIST[@]})))
fi
if [[ "$RUN_GATE_SWEEP" == "1" ]]; then
  total_jobs=$((total_jobs + ${#MAP_LIST[@]} * ${#SCENARIOS[@]} * ${#MODEL_LIST[@]} * ${#GATE_CONFIGS[@]}))
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
  local label_suffix="$3"
  local planner_mode="$4"
  local lm_model="$5"
  local feature_ablation="$6"
  local gate_variants="$7"
  local gate_votes="$8"
  local gate_consistency="$9"

  local case_file="$(case_file_for "$map" "$scenario")"
  local model_tag="$(sanitize "$lm_model")"
  local model_file="$(model_file_for_algo "$MAIN_ALGO")"
  local label="${map}_${scenario}_${MAIN_ALGO}_${planner_mode}_${label_suffix}_${model_tag}"
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
  PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
    --algo "$MAIN_ALGO" \
    --model "$model_file" \
    --cases "$case_file" \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --maps "$map" \
    --map-source "gazebo_3d_projection" \
    --lm-studio-url "$LM_URL" \
    --lm-model "$lm_model" \
    --planner-mode "$planner_mode" \
    --order-gate-variants "$gate_variants" \
    --order-gate-min-votes "$gate_votes" \
    --order-gate-min-consistency "$gate_consistency" \
    --candidate-feature-ablation "$feature_ablation" \
    --repair-invalid-route \
    --run-label "$label" \
    --episode-csv "$episode_csv" \
    --summary-csv "$summary_csv" \
    2>&1 | tee "$log"
}

print_header "Paper A order-gate threshold sweep"
echo "out_dir=$OUT_DIR episodes=$EPISODES timesteps=$TIMESTEPS seed=$SEED"
echo "maps=${MAP_LIST[*]} scenarios=${SCENARIOS[*]} models=${MODEL_LIST[*]}"
echo "gate_configs=${GATE_CONFIGS[*]}"
echo "run_reference_ablation=$RUN_REFERENCE_ABLATION run_gate_sweep=$RUN_GATE_SWEEP"
echo "total_jobs=$total_jobs"

if [[ "$RUN_REFERENCE_ABLATION" == "1" ]]; then
  print_header "Stage 1: case-matched reference ablations"
  for map in "${MAP_LIST[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      run_condition "$map" "$scenario" "graph" "no_llm" "no_llm" "none" 3 2 0.67
      for model in "${MODEL_LIST[@]}"; do
        run_condition "$map" "$scenario" "canonical" "llm_step" "$model" "none" 3 2 0.67
        run_condition "$map" "$scenario" "shuffle_order" "llm_step" "$model" "shuffle_order" 3 2 0.67
        run_condition "$map" "$scenario" "order_ensemble" "llm_step_order_ensemble" "$model" "none" 5 3 0.60
      done
    done
  done
fi

if [[ "$RUN_GATE_SWEEP" == "1" ]]; then
  print_header "Stage 2: consistency-gate threshold sweep"
  for map in "${MAP_LIST[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      for model in "${MODEL_LIST[@]}"; do
        for cfg in "${GATE_CONFIGS[@]}"; do
          local_name="${cfg%%:*}"
          rest="${cfg#*:}"
          gate_variants="${rest%%:*}"
          rest="${rest#*:}"
          gate_votes="${rest%%:*}"
          gate_consistency="${rest#*:}"
          run_condition "$map" "$scenario" "gate_${local_name}_v${gate_variants}_m${gate_votes}_c${gate_consistency}" \
            "llm_step_consistency_gate" "$model" "none" "$gate_variants" "$gate_votes" "$gate_consistency"
        done
      done
    done
  done
fi

print_header "Stage 3: summarize"
run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
  --out-csv "$OUT_DIR/order_gate_threshold_sweep_summary_with_ci.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

echo "Order-gate threshold sweep complete: $OUT_DIR"
