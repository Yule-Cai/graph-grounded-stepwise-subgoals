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
SEED="${SEED:-31}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
PRIMARY_MODEL="${PRIMARY_MODEL:-lfm2.5-8b-a1b-mlx}"
MODEL_LIST=(${=MODEL_LIST:-lfm2.5-8b-a1b-mlx nvidia/nemotron-3-nano-4b qwen/qwen3-1.7b google/gemma-3-1b liquid/lfm2.5-1.2b})
INCLUDE_HF_EXTERNAL_MODELS="${INCLUDE_HF_EXTERNAL_MODELS:-0}"
HF_EXTERNAL_MODELS_DIR="${HF_EXTERNAL_MODELS_DIR:-<LOCAL_SOURCE_PATH>"
SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
MAIN_ALGO="${MAIN_ALGO:-ppo}"
NO_LLM_ALGOS=(${=NO_LLM_ALGOS:-ppo sac})
CONTROLLER_ABLATION_ALGOS=(${=CONTROLLER_ABLATION_ALGOS:-sac})
PLANNER_MODES=(${=PLANNER_MODES:-llm_step llm_raw})
LLM_RETRIES="${LLM_RETRIES:-2}"
ORDER_GATE_VARIANTS="${ORDER_GATE_VARIANTS:-3}"
ORDER_GATE_MIN_VOTES="${ORDER_GATE_MIN_VOTES:-2}"
ORDER_GATE_MIN_CONSISTENCY="${ORDER_GATE_MIN_CONSISTENCY:-0.67}"
PARSE_REASONING_CONTENT="${PARSE_REASONING_CONTENT:-1}"
CANDIDATE_FEATURE_ABLATION="${CANDIDATE_FEATURE_ABLATION:-none}"
MAX_NO_LLM_PARALLEL="${MAX_NO_LLM_PARALLEL:-4}"
RUN_NO_LLM="${RUN_NO_LLM:-1}"
RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-1}"
RUN_CONTROLLER_ABLATION="${RUN_CONTROLLER_ABLATION:-1}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-1}"

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/final_aaai_stepwise_${STAMP}}"
MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"
REPORT_SCRIPT="$WORKSPACE/paper_assets/paper_a/write_final_experiment_report.py"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs"
SWEEP_START_TS="$(date +%s)"
JOB_DONE=0
BG_PIDS=()
BG_NAMES=()
FAILED_JOBS=()

reasoning_args=()
if [[ "$PARSE_REASONING_CONTENT" == "0" ]]; then
  reasoning_args=(--no-parse-reasoning-content)
fi

discover_hf_external_models() {
  local model_root="$1"
  local manifest="$2"
  if [[ ! -d "$model_root" ]]; then
    echo "HF external model directory not found: $model_root" > "$manifest"
    return 0
  fi
  python - "$model_root" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
allowed_arch = ("CausalLM", "LMHeadModel")
blocked_model_types = {"bert", "roberta", "xlm-roberta", "distilbert", "albert", "mobilebert"}
rows = []
for cfg in sorted(root.glob("*/config.json")):
    try:
        data = json.loads(cfg.read_text())
    except Exception as exc:
        rows.append({"path": str(cfg.parent), "include": False, "reason": f"config_read_error:{exc}"})
        continue
    arch = data.get("architectures") or []
    model_type = data.get("model_type") or ""
    is_causal = any(any(token in item for token in allowed_arch) for item in arch)
    include = is_causal and model_type not in blocked_model_types
    rows.append(
        {
            "path": str(cfg.parent),
            "name": cfg.parent.name,
            "model_type": model_type,
            "architectures": arch,
            "include": include,
            "reason": "causal_generation_model" if include else "not_a_causal_text_generator",
        }
    )
manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
for row in rows:
    if row["include"]:
        print(row["path"])
PY
}

dedupe_model_list() {
  local seen=" "
  local deduped=()
  local model
  for model in "${MODEL_LIST[@]}"; do
    if [[ "$seen" != *" $model "* ]]; then
      deduped+=("$model")
      seen="${seen}${model} "
    fi
  done
  MODEL_LIST=("${deduped[@]}")
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
  if [[ ! -f "$path" ]]; then
    echo 0
    return 0
  fi
  local lines=0
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    lines=$((lines + 1))
  done < "$path"
  if (( lines <= 1 )); then
    echo 0
  else
    echo $((lines - 1))
  fi
}

summary_completed_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return 0
  fi
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
  if (( total_jobs > 0 )); then
    filled=$((done_jobs * width / total_jobs))
  fi
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
  if (( total_jobs > 0 )); then
    percent=$((JOB_DONE * 100 / total_jobs))
  fi
  printf "%s %3d%% %s done=%d/%d elapsed=%s eta=%s\n" \
    "$(progress_bar "$JOB_DONE" "$total_jobs")" "$percent" "$phase" "$JOB_DONE" "$total_jobs" \
    "$(format_duration "$elapsed")" "$(format_duration "$eta")" | tee -a "$OUT_DIR/_status.log"
}

count_jobs() {
  local total=0
  if [[ "$RUN_NO_LLM" == "1" ]]; then
    total=$((total + ${#SCENARIOS} * ${#NO_LLM_ALGOS}))
  fi
  if [[ "$RUN_MODEL_SWEEP" == "1" ]]; then
    total=$((total + ${#MODEL_LIST} * ${#PLANNER_MODES} * ${#SCENARIOS}))
  fi
  if [[ "$RUN_CONTROLLER_ABLATION" == "1" ]]; then
    total=$((total + ${#CONTROLLER_ABLATION_ALGOS} * ${#SCENARIOS}))
  fi
  echo "$total"
}

run_map_job() {
  local scenario="$1"
  local algo="$2"
  local mode="$3"
  local lm_model="$4"
  local label_model="$5"
  local model_file
  model_file="$(model_file_for_algo "$algo")"
  if [[ -d "$label_model" ]]; then
    label_model="$(basename "$label_model")"
  fi
  local model_tag
  model_tag="$(sanitize "$label_model")"
  local cases="$ROOT/experiments/paper_a/cases/proactive_${scenario}_100.csv"
  local label="${scenario}_${algo}_${mode}_${model_tag}"
  if [[ "$SEED" != "31" ]]; then
    label="${label}_seed${SEED}"
  fi
  if [[ "$CANDIDATE_FEATURE_ABLATION" != "none" ]]; then
    label="${label}_${CANDIDATE_FEATURE_ABLATION}"
  fi
  local log="$OUT_DIR/${label}_${EPISODES}ep.log"
  local episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
  local summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"

  if [[ "$SKIP_COMPLETED" == "1" ]]; then
    local completed_rows
    completed_rows="$(summary_completed_rows "$summary_csv")"
    if (( completed_rows <= 0 )); then
      completed_rows="$(csv_data_rows "$episode_csv")"
    fi
    if (( completed_rows >= MIN_COMPLETED_ROWS )); then
      echo "SKIP completed scenario=$scenario algo=$algo mode=$mode model=$lm_model rows=$completed_rows summary_csv=$summary_csv episode_csv=$episode_csv" | tee -a "$OUT_DIR/_status.log"
      return 0
    fi
  fi

  echo "RUN scenario=$scenario algo=$algo mode=$mode model=$lm_model" | tee -a "$OUT_DIR/_status.log"
  PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
    --algo "$algo" \
    --model "$model_file" \
    --cases "$cases" \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --lm-studio-url "$LM_URL" \
    --lm-model "$lm_model" \
    --planner-mode "$mode" \
    --llm-retries "$LLM_RETRIES" \
    --order-gate-variants "$ORDER_GATE_VARIANTS" \
    --order-gate-min-votes "$ORDER_GATE_MIN_VOTES" \
    --order-gate-min-consistency "$ORDER_GATE_MIN_CONSISTENCY" \
    --candidate-feature-ablation "$CANDIDATE_FEATURE_ABLATION" \
    "${reasoning_args[@]}" \
    --repair-invalid-route \
    --run-label "$label" \
    --episode-csv "$episode_csv" \
    --summary-csv "$summary_csv" \
    2>&1 | tee "$log"
  local code="${pipestatus[1]}"
  if [[ "$code" -eq 0 ]]; then
    echo "OK scenario=$scenario algo=$algo mode=$mode model=$lm_model" | tee -a "$OUT_DIR/_status.log"
  else
    echo "FAIL scenario=$scenario algo=$algo mode=$mode model=$lm_model code=$code log=$log" | tee -a "$OUT_DIR/_status.log"
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
    if [[ "$code" -ne 0 ]]; then
      FAILED_JOBS+=("$name")
    fi
    BG_PIDS=(${BG_PIDS[2,-1]})
    BG_NAMES=(${BG_NAMES[2,-1]})
    print_progress "parallel job finished: $name" "$total_jobs"
  done
}

wait_all_parallel() {
  local total_jobs="$1"
  while (( ${#BG_PIDS} > 0 )); do
    local pid="${BG_PIDS[1]}"
    local name="${BG_NAMES[1]}"
    wait "$pid"
    local code="$?"
    JOB_DONE=$((JOB_DONE + 1))
    if [[ "$code" -ne 0 ]]; then
      FAILED_JOBS+=("$name")
    fi
    BG_PIDS=(${BG_PIDS[2,-1]})
    BG_NAMES=(${BG_NAMES[2,-1]})
    print_progress "parallel job finished: $name" "$total_jobs"
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
  if [[ "$code" -ne 0 ]]; then
    FAILED_JOBS+=("$name")
  fi
  print_progress "finished: $name" "$total_jobs"
}

write_summary_and_report() {
  local summary_csv="$OUT_DIR/final_experiment_summary_lmstudio_only.csv"
  local report_md="$OUT_DIR/final_experiment_report_lmstudio_only.md"
  if [[ "$INCLUDE_HF_EXTERNAL_MODELS" == "1" ]]; then
    summary_csv="$OUT_DIR/final_experiment_summary_with_ci.csv"
    report_md="$OUT_DIR/final_experiment_report.md"
  fi
  local episode_files=("$OUT_DIR"/episodes/*.csv(N))
  if (( ${#episode_files} == 0 )); then
    echo "No episode CSV files found; skipping summary and report." | tee -a "$OUT_DIR/_status.log"
    return 0
  fi
  run_py "$SUMMARY_SCRIPT" \
    --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
    --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
    --out-csv "$summary_csv" \
    2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

  run_py "$REPORT_SCRIPT" \
    --summary-csv "$summary_csv" \
    --run-dir "$OUT_DIR" \
    --out-md "$report_md" \
    --episodes "$EPISODES" \
    --timesteps "$TIMESTEPS" \
    --model-list "${MODEL_LIST[*]}" \
    --primary-model "$PRIMARY_MODEL" \
    2>&1 | tee "$OUT_DIR/stage_logs/final_report.log"
}

package_paper_a() {
  local package_stamp
  package_stamp="$(date +"%Y%m%d_%H%M%S")"
  local tmp_zip="/private/tmp/paper_a_final_bundle_${package_stamp}.zip"
  local final_zip="$OUT_DIR/paper_a_final_bundle_${package_stamp}.zip"
  echo "Packaging Paper A bundle..." | tee -a "$OUT_DIR/_status.log"
  (
    cd "$WORKSPACE"
    zip -qr "$tmp_zip" \
      paper_assets/paper_a \
      paper_a_experiments_desktop \
      results \
      -x '*/__pycache__/*' \
      -x '*.pyc' \
      -x 'paper_assets/paper_a/raw/*' \
      -x 'paper_assets/paper_a/tables/raw_*' \
      -x 'paper_assets/paper_a/rerun_logs/*/paper_a_final_bundle_*.zip' \
      -x 'paper_assets/paper_a/rerun_logs/*/episodes/*HuggingFace*' \
      -x 'paper_assets/paper_a/rerun_logs/*/summaries/*HuggingFace*' \
      -x 'paper_assets/paper_a/rerun_logs/*/paper_a_final_bundle_*.zip'
  )
  mv "$tmp_zip" "$final_zip"
  echo "$final_zip" > "$OUT_DIR/package_path.txt"
  echo "Package saved: $final_zip" | tee -a "$OUT_DIR/_status.log"

  local summary_csv="$OUT_DIR/final_experiment_summary_lmstudio_only.csv"
  local report_md="$OUT_DIR/final_experiment_report_lmstudio_only.md"
  if [[ "$INCLUDE_HF_EXTERNAL_MODELS" == "1" ]]; then
    summary_csv="$OUT_DIR/final_experiment_summary_with_ci.csv"
    report_md="$OUT_DIR/final_experiment_report.md"
  fi
  if [[ -f "$summary_csv" ]]; then
    run_py "$REPORT_SCRIPT" \
      --summary-csv "$summary_csv" \
      --run-dir "$OUT_DIR" \
      --out-md "$report_md" \
      --episodes "$EPISODES" \
      --timesteps "$TIMESTEPS" \
      --model-list "${MODEL_LIST[*]}" \
      --primary-model "$PRIMARY_MODEL" \
      --zip-path "$final_zip" \
      2>&1 | tee "$OUT_DIR/stage_logs/final_report_with_package.log"
  fi
}

if [[ "$INCLUDE_HF_EXTERNAL_MODELS" == "1" ]]; then
  HF_MANIFEST="$OUT_DIR/hf_external_models_manifest.json"
  HF_MODELS=($(discover_hf_external_models "$HF_EXTERNAL_MODELS_DIR" "$HF_MANIFEST"))
  if (( ${#HF_MODELS} > 0 )); then
    MODEL_LIST+=("${HF_MODELS[@]}")
    dedupe_model_list
  fi
else
  HF_MANIFEST="$OUT_DIR/hf_external_models_manifest.json"
  echo "HF external model discovery disabled." > "$HF_MANIFEST"
fi

TOTAL_JOBS="$(count_jobs)"

print_header "Paper A final AAAI stepwise experiments"
echo "episodes=$EPISODES timesteps=$TIMESTEPS lm_url=$LM_URL" | tee -a "$OUT_DIR/_status.log"
echo "seed=$SEED candidate_feature_ablation=$CANDIDATE_FEATURE_ABLATION" | tee -a "$OUT_DIR/_status.log"
echo "primary_model=$PRIMARY_MODEL" | tee -a "$OUT_DIR/_status.log"
echo "models=${MODEL_LIST[*]}" | tee -a "$OUT_DIR/_status.log"
echo "include_hf_external_models=$INCLUDE_HF_EXTERNAL_MODELS hf_external_models_dir=$HF_EXTERNAL_MODELS_DIR" | tee -a "$OUT_DIR/_status.log"
echo "hf_external_manifest=$HF_MANIFEST" | tee -a "$OUT_DIR/_status.log"
echo "skip_completed=$SKIP_COMPLETED" | tee -a "$OUT_DIR/_status.log"
echo "scenarios=${SCENARIOS[*]} main_algo=$MAIN_ALGO modes=${PLANNER_MODES[*]}" | tee -a "$OUT_DIR/_status.log"
echo "no_llm_algos=${NO_LLM_ALGOS[*]} controller_ablation_algos=${CONTROLLER_ABLATION_ALGOS[*]}" | tee -a "$OUT_DIR/_status.log"
echo "total_jobs=$TOTAL_JOBS out_dir=$OUT_DIR" | tee -a "$OUT_DIR/_status.log"

echo "model list at start:" > "$OUT_DIR/lmstudio_models_at_start.json"
curl -s "$LM_URL/v1/models" >> "$OUT_DIR/lmstudio_models_at_start.json" || true

if [[ "$RUN_NO_LLM" == "1" ]]; then
  print_header "Stage 1: no-LLM baselines (parallel)"
  for scenario in "${SCENARIOS[@]}"; do
    for algo in "${NO_LLM_ALGOS[@]}"; do
      name="no_llm/$scenario/$algo"
      wait_for_slot "$MAX_NO_LLM_PARALLEL" "$TOTAL_JOBS"
      run_map_job "$scenario" "$algo" "no_llm" "$PRIMARY_MODEL" "no_llm" &
      BG_PIDS+=("$!")
      BG_NAMES+=("$name")
    done
  done
  wait_all_parallel "$TOTAL_JOBS"
fi

if [[ "$RUN_MODEL_SWEEP" == "1" ]]; then
  print_header "Stage 2: LLM model sweep (sequential for LM Studio)"
  for model in "${MODEL_LIST[@]}"; do
    for mode in "${PLANNER_MODES[@]}"; do
      for scenario in "${SCENARIOS[@]}"; do
        name="$mode/$scenario/$MAIN_ALGO/$model"
        run_sequential_job "$TOTAL_JOBS" "$name" run_map_job "$scenario" "$MAIN_ALGO" "$mode" "$model" "$model"
      done
    done
  done
fi

if [[ "$RUN_CONTROLLER_ABLATION" == "1" ]]; then
  print_header "Stage 3: controller ablation for primary model"
  for algo in "${CONTROLLER_ABLATION_ALGOS[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      name="llm_step/$scenario/$algo/$PRIMARY_MODEL"
      run_sequential_job "$TOTAL_JOBS" "$name" run_map_job "$scenario" "$algo" "llm_step" "$PRIMARY_MODEL" "$PRIMARY_MODEL"
    done
  done
fi

print_header "Stage 4: summarize and report"
write_summary_and_report

if [[ "$RUN_PACKAGE" == "1" ]]; then
  print_header "Stage 5: package Paper A bundle"
  package_paper_a
fi

print_header "Finished"
echo "Outputs: $OUT_DIR"
if [[ "$INCLUDE_HF_EXTERNAL_MODELS" == "1" ]]; then
  echo "Summary: $OUT_DIR/final_experiment_summary_with_ci.csv"
  echo "Report: $OUT_DIR/final_experiment_report.md"
else
  echo "Summary: $OUT_DIR/final_experiment_summary_lmstudio_only.csv"
  echo "Report: $OUT_DIR/final_experiment_report_lmstudio_only.md"
fi
if [[ -f "$OUT_DIR/package_path.txt" ]]; then
  echo "Package: $(cat "$OUT_DIR/package_path.txt")"
fi
if (( ${#FAILED_JOBS} > 0 )); then
  echo "Failed jobs:"
  printf '  %s\n' "${FAILED_JOBS[@]}"
  exit 1
fi
