#!/usr/bin/env zsh
set -uo pipefail

WORKSPACE="${WORKSPACE:-$(pwd)}"
DESKTOP_ROOT="${DESKTOP_ROOT:-$WORKSPACE/paper_a_experiments_desktop}"
source "$DESKTOP_ROOT/scripts/_paper_a_common.sh"

EPISODES="${1:-30}"
TIMESTEPS="${2:-5000000}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/model_sweep_${STAMP}}"

MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"

SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
ALGOS=(${=ALGOS:-ppo})
PLANNER_MODES=(${=PLANNER_MODES:-llm_retry})
LLM_RETRIES="${LLM_RETRIES:-2}"
PARSE_REASONING_CONTENT="${PARSE_REASONING_CONTENT:-1}"
MODELS=(${=MODEL_LIST:-lfm2.5-8b-a1b-mlx nvidia/nemotron-3-nano-4b qwen/qwen3-1.7b google/gemma-3-1b liquid/lfm2.5-1.2b google/gemma-4-e4b})
reasoning_args=()
if [[ "$PARSE_REASONING_CONTENT" == "0" ]]; then
  reasoning_args=(--no-parse-reasoning-content)
fi

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs"
TOTAL_JOBS=$((${#MODELS} * ${#PLANNER_MODES} * ${#SCENARIOS} * ${#ALGOS}))
JOB_INDEX=0
SWEEP_START_TS="$(date +%s)"

print_header "Paper A local-model sweep"
echo "episodes=$EPISODES timesteps=$TIMESTEPS lm_url=$LM_URL"
echo "models=${MODELS[*]}"
echo "scenarios=${SCENARIOS[*]} algos=${ALGOS[*]} modes=${PLANNER_MODES[*]}"
echo "total_jobs=$TOTAL_JOBS"
echo "out_dir=$OUT_DIR"

sanitize() {
  echo "$1" | sed 's#[/: ]#_#g' | sed 's#[^A-Za-z0-9_.-]#_#g'
}

model_file_for_algo() {
  local algo="$1"
  case "$algo" in
    ppo) echo "models/PPO/ppo_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    sac) echo "models/SAC/sac_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    *) echo "ERROR_UNKNOWN_ALGO" ;;
  esac
}

format_duration() {
  local seconds="$1"
  if (( seconds < 0 )); then
    seconds=0
  fi
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
  local width=28
  local filled=0
  if (( total_jobs > 0 )); then
    filled=$((done_jobs * width / total_jobs))
  fi
  local empty=$((width - filled))
  printf "["
  local i
  for (( i = 0; i < filled; i++ )); do
    printf "#"
  done
  for (( i = 0; i < empty; i++ )); do
    printf "-"
  done
  printf "]"
}

print_progress() {
  local phase="$1"
  local done_jobs="$2"
  local total_jobs="$3"
  local start_ts="$4"
  local now_ts="$(date +%s)"
  local elapsed=$((now_ts - start_ts))
  local eta=0
  if (( done_jobs > 0 && total_jobs > done_jobs )); then
    eta=$(((elapsed * (total_jobs - done_jobs)) / done_jobs))
  fi
  local percent=0
  if (( total_jobs > 0 )); then
    percent=$((done_jobs * 100 / total_jobs))
  fi
  printf "%s " "$(progress_bar "$done_jobs" "$total_jobs")"
  printf "%3d%% %s %d/%d elapsed=%s eta=%s\n" \
    "$percent" "$phase" "$done_jobs" "$total_jobs" "$(format_duration "$elapsed")" "$(format_duration "$eta")"
}

echo "model list at start:" > "$OUT_DIR/lmstudio_models_at_start.json"
curl -s "$LM_URL/v1/models" >> "$OUT_DIR/lmstudio_models_at_start.json" || true

for model in "${MODELS[@]}"; do
  model_tag="$(sanitize "$model")"
  for mode in "${PLANNER_MODES[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      for algo in "${ALGOS[@]}"; do
        JOB_INDEX=$((JOB_INDEX + 1))
        model_file="$(model_file_for_algo "$algo")"
        cases="$ROOT/experiments/paper_a/cases/proactive_${scenario}_100.csv"
        label="${scenario}_${algo}_${mode}_${model_tag}"
        log="$OUT_DIR/${label}_${EPISODES}ep.log"
        episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
        summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"

        print_progress "starting" $((JOB_INDEX - 1)) "$TOTAL_JOBS" "$SWEEP_START_TS" | tee -a "$OUT_DIR/_status.log"
        echo "RUN job=$JOB_INDEX/$TOTAL_JOBS model=$model scenario=$scenario algo=$algo mode=$mode log=$log" | tee -a "$OUT_DIR/_status.log"
        job_start_ts="$(date +%s)"
        PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
          --algo "$algo" \
          --model "$model_file" \
          --cases "$cases" \
          --episodes "$EPISODES" \
          --lm-studio-url "$LM_URL" \
          --lm-model "$model" \
          --planner-mode "$mode" \
          --llm-retries "$LLM_RETRIES" \
          "${reasoning_args[@]}" \
          --repair-invalid-route \
          --run-label "$label" \
          --episode-csv "$episode_csv" \
          --summary-csv "$summary_csv" \
          2>&1 | tee "$log"
        code="${pipestatus[1]}"
        job_elapsed=$(($(date +%s) - job_start_ts))
        if [[ "$code" -eq 0 ]]; then
          echo "OK job=$JOB_INDEX/$TOTAL_JOBS model=$model scenario=$scenario algo=$algo mode=$mode job_elapsed=$(format_duration "$job_elapsed")" | tee -a "$OUT_DIR/_status.log"
        else
          echo "FAIL model=$model scenario=$scenario algo=$algo mode=$mode code=$code log=$log" | tee -a "$OUT_DIR/_status.log"
          tail -n 40 "$log"
        fi
        print_progress "finished" "$JOB_INDEX" "$TOTAL_JOBS" "$SWEEP_START_TS" | tee -a "$OUT_DIR/_status.log"
      done
    done
  done
done

episode_files=("$OUT_DIR"/episodes/*.csv(N))
if (( ${#episode_files} > 0 )); then
  run_py "$SUMMARY_SCRIPT" \
    --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
    --out-csv "$OUT_DIR/model_sweep_summary_with_ci.csv" \
    2>&1 | tee "$OUT_DIR/stage_logs/model_sweep_summary.log"
fi

print_header "Finished model sweep"
echo "All outputs saved under: $OUT_DIR"
echo "Combined summary: $OUT_DIR/model_sweep_summary_with_ci.csv"
