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

SAMPLES="${1:-50}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/latency_${STAMP}}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
MODEL_LIST=(${=MODEL_LIST:-liquid/lfm2.5-1.2b nvidia/nemotron-3-nano-4b})
MAP_LIST=(${=MAP_LIST:-reference_family_flat reference_villa_ground studio_apartment})
SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
CASE_DIR="${CASE_DIR:-$DESKTOP_ROOT/experiments/paper_a/cases/multimap_generalization}"
GENERATE_CASES="${GENERATE_CASES:-1}"
MAP_SOURCE="${MAP_SOURCE:-gazebo_3d_projection}"
EPISODES_FOR_CASES="${EPISODES_FOR_CASES:-100}"

CASE_SCRIPT="$WORKSPACE/paper_assets/paper_a/generate_multimap_cases.py"
LATENCY_SCRIPT="$WORKSPACE/paper_assets/paper_a/measure_llm_step_latency.py"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs" "$CASE_DIR"

sanitize() {
  echo "$1" | sed 's#[/: ]#_#g' | sed 's#[^A-Za-z0-9_.-]#_#g'
}

print_header "Paper A edge-LLM latency measurements"
echo "out_dir=$OUT_DIR" | tee -a "$OUT_DIR/_status.log"
echo "samples=$SAMPLES lm_url=$LM_URL models=${MODEL_LIST[*]}" | tee -a "$OUT_DIR/_status.log"
echo "maps=${MAP_LIST[*]} scenarios=${SCENARIOS[*]}" | tee -a "$OUT_DIR/_status.log"

if [[ "$GENERATE_CASES" == "1" ]]; then
  print_header "Stage 0: ensure latency case CSVs exist"
  run_py "$CASE_SCRIPT" \
    --maps "${MAP_LIST[*]}" \
    --scenarios "${SCENARIOS[*]}" \
    --episodes "$EPISODES_FOR_CASES" \
    --out-dir "$CASE_DIR" \
    2>&1 | tee "$OUT_DIR/stage_logs/generate_cases.log"
fi

total=$(( ${#MODEL_LIST} * ${#MAP_LIST} * ${#SCENARIOS} ))
done_jobs=0
for model in "${MODEL_LIST[@]}"; do
  model_tag="$(sanitize "$model")"
  for map_id in "${MAP_LIST[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
      done_jobs=$((done_jobs + 1))
      case_file="$CASE_DIR/proactive_${map_id}_${scenario}_${EPISODES_FOR_CASES}.csv"
      raw_csv="$OUT_DIR/raw/latency_${model_tag}_${map_id}_${scenario}_${SAMPLES}.csv"
      summary_csv="$OUT_DIR/summaries/latency_${model_tag}_${map_id}_${scenario}_${SAMPLES}.csv"
      echo "[$done_jobs/$total] latency model=$model map=$map_id scenario=$scenario" | tee -a "$OUT_DIR/_status.log"
      run_py "$LATENCY_SCRIPT" \
        --cases "$case_file" \
        --samples "$SAMPLES" \
        --maps "$map_id" \
        --map-source "$MAP_SOURCE" \
        --lm-studio-url "$LM_URL" \
        --lm-model "$model" \
        --out-csv "$raw_csv" \
        --summary-csv "$summary_csv" \
        2>&1 | tee "$OUT_DIR/latency_${model_tag}_${map_id}_${scenario}.log"
    done
  done
done

python - "$OUT_DIR" <<'PY'
import csv, glob, os, sys
out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "summaries", "*.csv"))):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["source_file"] = os.path.basename(path)
            rows.append(row)
if not rows:
    raise SystemExit("No latency summaries were produced")
path = os.path.join(out_dir, "latency_summary_all.csv")
with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(path)
PY

print_header "Finished"
echo "Output directory: $OUT_DIR"
echo "Summary: $OUT_DIR/latency_summary_all.csv"
