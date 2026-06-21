#!/usr/bin/env zsh
set -uo pipefail

WORKSPACE="${WORKSPACE:-$(pwd)}"
DESKTOP_ROOT="${DESKTOP_ROOT:-$WORKSPACE/paper_a_experiments_desktop}"
source "$DESKTOP_ROOT/scripts/_paper_a_common.sh"

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
LM_MODEL="${LM_MODEL:-lfm2.5-8b-a1b-mlx}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/a1b_mlx_all_${STAMP}}"

MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"
ALIGNMENT_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_route_preference_alignment.py"
PROACTIVE_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_lfm25_a1b_mlx_proactive_rerun.zsh"

SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
ALGOS=(${=ALGOS:-ppo sac})
LLM_MODES=(${=LLM_MODES:-llm llm_retry})
LLM_RETRIES="${LLM_RETRIES:-2}"
MAX_NO_LLM_PARALLEL="${MAX_NO_LLM_PARALLEL:-4}"
RUN_NO_LLM="${RUN_NO_LLM:-1}"
RUN_LLM_QUEUE="${RUN_LLM_QUEUE:-1}"
RUN_ALIGNMENT="${RUN_ALIGNMENT:-1}"
RUN_PROACTIVE_DIAGNOSTIC="${RUN_PROACTIVE_DIAGNOSTIC:-1}"
ALIGNMENT_LIMIT="${ALIGNMENT_LIMIT:-20}"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs"
BG_PIDS=()

print_header "Paper A all A1B-MLX reruns"
echo "episodes=$EPISODES timesteps=$TIMESTEPS"
echo "lm_model=$LM_MODEL lm_url=$LM_URL"
echo "scenarios=${SCENARIOS[*]} algos=${ALGOS[*]}"
echo "llm_modes=${LLM_MODES[*]} llm_retries=$LLM_RETRIES"
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

run_map_job() {
  local scenario="$1"
  local algo="$2"
  local mode="$3"
  local model_file
  model_file="$(model_file_for_algo "$algo")"
  local cases="$ROOT/experiments/paper_a/cases/proactive_${scenario}_100.csv"
  local label="${scenario}_${algo}_${mode}"
  local log="$OUT_DIR/${label}_${EPISODES}ep.log"
  local episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
  local summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"

  echo "RUN scenario=$scenario algo=$algo mode=$mode" | tee -a "$OUT_DIR/_status.log"
  run_py "$MAP_SCRIPT" \
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
  local code="$?"
  if [[ "$code" -eq 0 ]]; then
    echo "OK scenario=$scenario algo=$algo mode=$mode" | tee -a "$OUT_DIR/_status.log"
    tail -n 18 "$log"
  else
    echo "FAIL scenario=$scenario algo=$algo mode=$mode code=$code log=$log" | tee -a "$OUT_DIR/_status.log"
    tail -n 40 "$log"
  fi
  return "$code"
}

wait_for_slot() {
  local limit="$1"
  while (( ${#BG_PIDS} >= limit )); do
    local pid="${BG_PIDS[1]}"
    wait "$pid"
    local code="$?"
    if [[ "$code" -ne 0 ]]; then
      echo "Background job failed pid=$pid code=$code" | tee -a "$OUT_DIR/_status.log"
    fi
    BG_PIDS=(${BG_PIDS[2,-1]})
  done
}

wait_all_jobs() {
  local failed=0
  while (( ${#BG_PIDS} > 0 )); do
    local pid="${BG_PIDS[1]}"
    wait "$pid"
    local code="$?"
    if [[ "$code" -ne 0 ]]; then
      failed=1
    fi
    BG_PIDS=(${BG_PIDS[2,-1]})
  done
  return "$failed"
}

echo "model list at start:" > "$OUT_DIR/lmstudio_models_at_start.json"
curl -s "$LM_URL/v1/models" >> "$OUT_DIR/lmstudio_models_at_start.json" || true

if [[ "$RUN_NO_LLM" == "1" ]]; then
  print_header "Stage 1: no-LLM graph-search baselines (parallel)"
  for scenario in "${SCENARIOS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      wait_for_slot "$MAX_NO_LLM_PARALLEL"
      run_map_job "$scenario" "$algo" "no_llm" &
      BG_PIDS+=("$!")
    done
  done
  wait_all_jobs || {
    echo "At least one no-LLM job failed. Continuing to LLM queue; inspect _status.log." | tee -a "$OUT_DIR/_status.log"
  }
fi

if [[ "$RUN_LLM_QUEUE" == "1" ]]; then
  print_header "Stage 2: LLM map-conditioned queue (sequential)"
  for mode in "${LLM_MODES[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      for algo in "${ALGOS[@]}"; do
        run_map_job "$scenario" "$algo" "$mode" || {
          echo "LLM job failed; continuing with remaining jobs." | tee -a "$OUT_DIR/_status.log"
        }
      done
    done
  done
fi

print_header "Stage 3: summarize map-conditioned revision runs"
episode_files=("$OUT_DIR"/episodes/*.csv(N))
if (( ${#episode_files} > 0 )); then
  run_py "$SUMMARY_SCRIPT" \
    --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
    --out-csv "$OUT_DIR/revision_experiment_summary_with_ci.csv" \
    2>&1 | tee "$OUT_DIR/stage_logs/revision_summary.log"
else
  echo "No episode CSV files found; skipping revision summary." | tee -a "$OUT_DIR/_status.log"
fi

if [[ "$RUN_ALIGNMENT" == "1" ]]; then
  print_header "Stage 4: route-preference alignment diagnostic (LLM, sequential)"
  local_model_tag="$(sanitize "$LM_MODEL")"
  run_py "$ALIGNMENT_SCRIPT" \
    --model "$LM_MODEL" \
    --lm-studio-url "$LM_URL" \
    --limit "$ALIGNMENT_LIMIT" \
    --output "$OUT_DIR/route_preference_alignment_${local_model_tag}.csv" \
    2>&1 | tee "$OUT_DIR/stage_logs/route_preference_alignment.log"
fi

if [[ "$RUN_PROACTIVE_DIAGNOSTIC" == "1" ]]; then
  print_header "Stage 5: route-option proactive diagnostic rerun (LLM, sequential)"
  LOG_DIR="$OUT_DIR/proactive_diagnostic" \
  LM_MODEL="$LM_MODEL" \
  LM_STUDIO_URL="$LM_URL" \
  PAPER_A_SCENARIOS="${SCENARIOS[*]}" \
  PAPER_A_ALGOS="${ALGOS[*]}" \
  zsh "$PROACTIVE_SCRIPT" "$EPISODES" "$TIMESTEPS" \
    2>&1 | tee "$OUT_DIR/stage_logs/proactive_diagnostic.log"
fi

print_header "Finished"
echo "All outputs saved under: $OUT_DIR"
echo "Primary combined summary: $OUT_DIR/revision_experiment_summary_with_ci.csv"
