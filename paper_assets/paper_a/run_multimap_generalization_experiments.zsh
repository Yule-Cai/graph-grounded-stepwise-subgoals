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
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/multimap_generalization_${STAMP}}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
MAP_LIST=(${=MAP_LIST:-reference_villa_ground studio_apartment townhouse_long luxury_villa})
SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
SEEDS=(${=SEEDS:-31})
TOP_MODELS=(${=TOP_MODELS:-liquid/lfm2.5-1.2b nvidia/nemotron-3-nano-4b})
RAW_MODELS=(${=RAW_MODELS:-liquid/lfm2.5-1.2b})
RUN_NO_LLM="${RUN_NO_LLM:-1}"
RUN_LLM_STEP="${RUN_LLM_STEP:-1}"
RUN_LLM_RAW="${RUN_LLM_RAW:-1}"
RUN_ORDER_ENSEMBLE="${RUN_ORDER_ENSEMBLE:-1}"
RUN_SHUFFLE_ENSEMBLE="${RUN_SHUFFLE_ENSEMBLE:-1}"
GENERATE_CASES="${GENERATE_CASES:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-$EPISODES}"
MAX_NO_LLM_PARALLEL="${MAX_NO_LLM_PARALLEL:-4}"
MAP_SOURCE="${MAP_SOURCE:-gazebo_3d_projection}"
MAIN_ALGO="${MAIN_ALGO:-ppo}"

MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
CASE_SCRIPT="$WORKSPACE/paper_assets/paper_a/generate_multimap_cases.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"
REPORT_SCRIPT="$WORKSPACE/paper_assets/paper_a/write_final_experiment_report.py"
BOOTSTRAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/bootstrap_paper_a_ci.py"
CASE_DIR="$DESKTOP_ROOT/experiments/paper_a/cases/multimap_generalization"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs" "$CASE_DIR"
SWEEP_START_TS="$(date +%s)"
JOB_DONE=0
BG_PIDS=()
BG_NAMES=()
FAILED_JOBS=()

sanitize() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

format_duration() {
  local seconds="$1"
  if (( seconds < 0 )); then seconds=0; fi
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))
  if (( hours > 0 )); then
    printf "%dh%02dm%02ds" "$hours" "$minutes" "$secs"
  else
    printf "%dm%02ds" "$minutes" "$secs"
  fi
}

progress_bar() {
  local done_jobs="$1"
  local total_jobs="$2"
  local width=32
  local filled=0
  if (( total_jobs > 0 )); then filled=$((done_jobs * width / total_jobs)); fi
  local empty=$((width - filled))
  printf "["
  local i
  for (( i = 0; i < filled; i++ )); do printf "#"; done
  for (( i = 0; i < empty; i++ )); do printf "-"; done
  printf "]"
}

print_progress() {
  local phase="$1"
  local total_jobs="$2"
  local now_ts="$(date +%s)"
  local elapsed=$((now_ts - SWEEP_START_TS))
  local eta=0
  if (( JOB_DONE > 0 && total_jobs > JOB_DONE )); then
    eta=$(((elapsed * (total_jobs - JOB_DONE)) / JOB_DONE))
  fi
  local percent=0
  if (( total_jobs > 0 )); then percent=$((JOB_DONE * 100 / total_jobs)); fi
  printf "%s %3d%% %s done=%d/%d elapsed=%s eta=%s\n" \
    "$(progress_bar "$JOB_DONE" "$total_jobs")" "$percent" "$phase" "$JOB_DONE" "$total_jobs" \
    "$(format_duration "$elapsed")" "$(format_duration "$eta")" | tee -a "$OUT_DIR/_status.log"
}

csv_data_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local lines=0
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do lines=$((lines + 1)); done < "$path"
  if (( lines <= 1 )); then echo 0; else echo $((lines - 1)); fi
}

summary_completed_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local header=""
  local row=""
  {
    IFS= read -r header
    IFS= read -r row
  } < "$path"
  if [[ -z "$header" || -z "$row" ]]; then echo 0; return 0; fi
  local -a header_cols row_cols
  header_cols=("${(@s:,:)header}")
  row_cols=("${(@s:,:)row}")
  local i key value
  for (( i = 1; i <= ${#header_cols}; i++ )); do
    key="${header_cols[$i]//\"/}"
    if [[ "$key" == "episodes" ]]; then
      value="${row_cols[$i]//\"/}"
      value="${value%%.*}"
      if [[ "$value" == <-> ]]; then
        echo "$value"
      else
        echo 0
      fi
      return 0
    fi
  done
  echo 0
}

model_file_for_algo() {
  local algo="$1"
  case "$algo" in
    ppo) echo "models/PPO/ppo_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    sac) echo "models/SAC/sac_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    *) echo "ERROR_UNKNOWN_ALGO" ;;
  esac
}

count_jobs() {
  local total=0
  local maps_count=${#MAP_LIST}
  local scen_count=${#SCENARIOS}
  local seed_count=${#SEEDS}
  if [[ "$RUN_NO_LLM" == "1" ]]; then total=$((total + maps_count * scen_count * seed_count)); fi
  if [[ "$RUN_LLM_STEP" == "1" ]]; then total=$((total + maps_count * scen_count * seed_count * ${#TOP_MODELS})); fi
  if [[ "$RUN_LLM_RAW" == "1" ]]; then total=$((total + maps_count * scen_count * seed_count * ${#RAW_MODELS})); fi
  if [[ "$RUN_ORDER_ENSEMBLE" == "1" ]]; then total=$((total + maps_count * scen_count * seed_count * ${#TOP_MODELS})); fi
  if [[ "$RUN_SHUFFLE_ENSEMBLE" == "1" ]]; then total=$((total + maps_count * scen_count * seed_count * ${#TOP_MODELS})); fi
  echo "$total"
}

run_job() {
  local map_id="$1"
  local scenario="$2"
  local seed="$3"
  local mode="$4"
  local lm_model="$5"
  local label_model="$6"
  local ablation="${7:-none}"
  local model_file
  model_file="$(model_file_for_algo "$MAIN_ALGO")"
  local model_tag
  model_tag="$(sanitize "$label_model")"
  local ablation_tag=""
  if [[ "$ablation" != "none" ]]; then ablation_tag="_${ablation}"; fi
  local case_file="$CASE_DIR/proactive_${map_id}_${scenario}_${EPISODES}.csv"
  local label="${map_id}_${scenario}_${MAIN_ALGO}_${mode}_${model_tag}_seed${seed}${ablation_tag}"
  local log="$OUT_DIR/${label}_${EPISODES}ep.log"
  local episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
  local summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"

  if [[ "$SKIP_COMPLETED" == "1" ]]; then
    local completed_rows
    completed_rows="$(summary_completed_rows "$summary_csv")"
    if (( completed_rows <= 0 )); then completed_rows="$(csv_data_rows "$episode_csv")"; fi
    if (( completed_rows >= MIN_COMPLETED_ROWS )); then
      echo "SKIP completed map=$map_id scenario=$scenario seed=$seed mode=$mode model=$lm_model rows=$completed_rows" | tee -a "$OUT_DIR/_status.log"
      return 0
    fi
  fi

  echo "RUN map=$map_id scenario=$scenario seed=$seed mode=$mode model=$lm_model ablation=$ablation" | tee -a "$OUT_DIR/_status.log"
  PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
    --algo "$MAIN_ALGO" \
    --model "$model_file" \
    --cases "$case_file" \
    --episodes "$EPISODES" \
    --seed "$seed" \
    --maps "$map_id" \
    --map-source "$MAP_SOURCE" \
    --lm-studio-url "$LM_URL" \
    --lm-model "$lm_model" \
    --planner-mode "$mode" \
    --candidate-feature-ablation "$ablation" \
    --repair-invalid-route \
    --run-label "$label" \
    --episode-csv "$episode_csv" \
    --summary-csv "$summary_csv" \
    2>&1 | tee "$log"
  local code="${pipestatus[1]}"
  if [[ "$code" -eq 0 ]]; then
    echo "OK map=$map_id scenario=$scenario seed=$seed mode=$mode model=$lm_model" | tee -a "$OUT_DIR/_status.log"
  else
    echo "FAIL map=$map_id scenario=$scenario seed=$seed mode=$mode model=$lm_model code=$code log=$log" | tee -a "$OUT_DIR/_status.log"
  fi
  return "$code"
}

wait_for_slot() {
  local limit="$1"
  local total_jobs="$2"
  while (( ${#BG_PIDS} >= limit )); do
    local pid="${BG_PIDS[1]}"
    local name="${BG_NAMES[1]}"
    wait "$pid"
    local code="$?"
    JOB_DONE=$((JOB_DONE + 1))
    if [[ "$code" -ne 0 ]]; then FAILED_JOBS+=("$name"); fi
    BG_PIDS=(${BG_PIDS[2,-1]})
    BG_NAMES=(${BG_NAMES[2,-1]})
    print_progress "parallel job finished: $name" "$total_jobs"
  done
}

wait_all_parallel() {
  local total_jobs="$1"
  while (( ${#BG_PIDS} > 0 )); do
    wait_for_slot 1 "$total_jobs"
  done
}

run_sequential_job() {
  local total_jobs="$1"
  local name="$2"
  shift 2
  print_progress "starting: $name" "$total_jobs"
  "$@"
  local code="$?"
  JOB_DONE=$((JOB_DONE + 1))
  if [[ "$code" -ne 0 ]]; then FAILED_JOBS+=("$name"); fi
  print_progress "finished: $name" "$total_jobs"
}

TOTAL_JOBS="$(count_jobs)"

print_header "Paper A multi-map generalization experiments"
echo "out_dir=$OUT_DIR" | tee -a "$OUT_DIR/_status.log"
echo "episodes=$EPISODES timesteps=$TIMESTEPS algo=$MAIN_ALGO lm_url=$LM_URL" | tee -a "$OUT_DIR/_status.log"
echo "maps=${MAP_LIST[*]} scenarios=${SCENARIOS[*]} seeds=${SEEDS[*]}" | tee -a "$OUT_DIR/_status.log"
echo "top_models=${TOP_MODELS[*]} raw_models=${RAW_MODELS[*]}" | tee -a "$OUT_DIR/_status.log"
echo "total_jobs=$TOTAL_JOBS" | tee -a "$OUT_DIR/_status.log"

if [[ "$GENERATE_CASES" == "1" ]]; then
  print_header "Stage 0: generate multi-map cases"
  run_py "$CASE_SCRIPT" \
    --maps "${MAP_LIST[*]}" \
    --scenarios "${SCENARIOS[*]}" \
    --episodes "$EPISODES" \
    --out-dir "$CASE_DIR" \
    2>&1 | tee "$OUT_DIR/stage_logs/generate_cases.log"
fi

if [[ "$RUN_NO_LLM" == "1" ]]; then
  print_header "Stage 1: risk-weighted no-LLM graph route (parallel)"
  for seed in "${SEEDS[@]}"; do
    for map_id in "${MAP_LIST[@]}"; do
      for scenario in "${SCENARIOS[@]}"; do
        local_name="no_llm/$map_id/$scenario/seed${seed}"
        wait_for_slot "$MAX_NO_LLM_PARALLEL" "$TOTAL_JOBS"
        run_job "$map_id" "$scenario" "$seed" "no_llm" "no_llm" "no_llm" "none" &
        BG_PIDS+=("$!")
        BG_NAMES+=("$local_name")
      done
    done
  done
  wait_all_parallel "$TOTAL_JOBS"
fi

if [[ "$RUN_LLM_STEP" == "1" ]]; then
  print_header "Stage 2: LLM stepwise top models (sequential for LM Studio)"
  for seed in "${SEEDS[@]}"; do
    for model in "${TOP_MODELS[@]}"; do
      for map_id in "${MAP_LIST[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
          run_sequential_job "$TOTAL_JOBS" "llm_step/$model/$map_id/$scenario/seed${seed}" run_job "$map_id" "$scenario" "$seed" "llm_step" "$model" "$model" "none"
        done
      done
    done
  done
fi

if [[ "$RUN_LLM_RAW" == "1" ]]; then
  print_header "Stage 3: raw full-route baseline (sequential for LM Studio)"
  for seed in "${SEEDS[@]}"; do
    for model in "${RAW_MODELS[@]}"; do
      for map_id in "${MAP_LIST[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
          run_sequential_job "$TOTAL_JOBS" "llm_raw/$model/$map_id/$scenario/seed${seed}" run_job "$map_id" "$scenario" "$seed" "llm_raw" "$model" "$model" "none"
        done
      done
    done
  done
fi

if [[ "$RUN_ORDER_ENSEMBLE" == "1" ]]; then
  print_header "Stage 4: order ensemble mitigation (canonical variants)"
  for seed in "${SEEDS[@]}"; do
    for model in "${TOP_MODELS[@]}"; do
      for map_id in "${MAP_LIST[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
          run_sequential_job "$TOTAL_JOBS" "order_ensemble/$model/$map_id/$scenario/seed${seed}" run_job "$map_id" "$scenario" "$seed" "llm_step_order_ensemble" "$model" "$model" "none"
        done
      done
    done
  done
fi

if [[ "$RUN_SHUFFLE_ENSEMBLE" == "1" ]]; then
  print_header "Stage 5: shuffled-order ensemble mitigation"
  for seed in "${SEEDS[@]}"; do
    for model in "${TOP_MODELS[@]}"; do
      for map_id in "${MAP_LIST[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
          run_sequential_job "$TOTAL_JOBS" "shuffle_ensemble/$model/$map_id/$scenario/seed${seed}" run_job "$map_id" "$scenario" "$seed" "llm_step_order_ensemble" "$model" "$model" "shuffle_order"
        done
      done
    done
  done
fi

print_header "Stage 6: summarize"
run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
  --out-csv "$OUT_DIR/multimap_generalization_summary_with_ci.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

run_py "$BOOTSTRAP_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --out-csv "$OUT_DIR/multimap_generalization_bootstrap_ci.csv" \
  --bootstrap-samples 2000 \
  2>&1 | tee "$OUT_DIR/stage_logs/bootstrap_ci.log"

run_py "$BOOTSTRAP_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --out-csv "$OUT_DIR/multimap_generalization_bootstrap_ci_by_map.csv" \
  --bootstrap-samples 2000 \
  --group-by-map \
  2>&1 | tee "$OUT_DIR/stage_logs/bootstrap_ci_by_map.log"

run_py "$REPORT_SCRIPT" \
  --summary-csv "$OUT_DIR/multimap_generalization_summary_with_ci.csv" \
  --run-dir "$OUT_DIR" \
  --out-md "$OUT_DIR/multimap_generalization_report.md" \
  --episodes "$EPISODES" \
  --timesteps "$TIMESTEPS" \
  --model-list "${TOP_MODELS[*]}" \
  --primary-model "${TOP_MODELS[1]}" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_report.log"

print_header "Finished"
echo "Output directory: $OUT_DIR"
echo "Summary: $OUT_DIR/multimap_generalization_summary_with_ci.csv"
echo "Bootstrap CI: $OUT_DIR/multimap_generalization_bootstrap_ci.csv"
echo "Report: $OUT_DIR/multimap_generalization_report.md"

if (( ${#FAILED_JOBS} > 0 )); then
  echo "Failed jobs:"
  printf '  %s\n' "${FAILED_JOBS[@]}"
  exit 1
fi
