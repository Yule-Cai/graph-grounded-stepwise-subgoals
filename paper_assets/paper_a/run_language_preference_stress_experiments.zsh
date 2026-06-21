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

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/language_preference_stress_${STAMP}}"
BASE_LIMIT="${BASE_LIMIT:-20}"
VARIANTS_PER_TASK="${VARIANTS_PER_TASK:-2}"
BENCHMARK="${BENCHMARK:-all}"
OPTION_ORDER="${OPTION_ORDER:-shuffled}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
MODEL_LIST=(${=MODEL_LIST:-nvidia/nemotron-3-nano-4b google/gemma-4-e4b liquid/lfm2.5-1.2b})
WRITE_PROMPTS_ONLY="${WRITE_PROMPTS_ONLY:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TIMEOUT_S="${TIMEOUT_S:-60}"

SCRIPT="$WORKSPACE/paper_assets/paper_a/run_language_preference_stress.py"
mkdir -p "$OUT_DIR/results" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs"

sanitize() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

print_header "Paper A language-preference stress diagnostic"
echo "out_dir=$OUT_DIR"
echo "base_limit=$BASE_LIMIT variants_per_task=$VARIANTS_PER_TASK benchmark=$BENCHMARK option_order=$OPTION_ORDER"
echo "lm_url=$LM_URL models=${MODEL_LIST[*]} write_prompts_only=$WRITE_PROMPTS_ONLY"

for model in "${MODEL_LIST[@]}"; do
  tag="$(sanitize "$model")"
  result_csv="$OUT_DIR/results/${tag}_language_preference_stress.csv"
  summary_csv="$OUT_DIR/summaries/${tag}_language_preference_stress_summary.csv"
  log="$OUT_DIR/stage_logs/${tag}.log"
  args=(
    "$SCRIPT"
    --cases "$WORKSPACE/paper_assets/paper_a/raw/preference_alignment_cases.csv"
    --output "$result_csv"
    --summary-output "$summary_csv"
    --limit "$BASE_LIMIT"
    --benchmark "$BENCHMARK"
    --variants-per-task "$VARIANTS_PER_TASK"
    --option-order "$OPTION_ORDER"
    --lm-studio-url "$LM_URL"
    --model "$model"
    --timeout-s "$TIMEOUT_S"
    --max-tokens "$MAX_TOKENS"
  )
  if [[ "$WRITE_PROMPTS_ONLY" == "1" ]]; then
    args+=(--write-prompts-only)
  fi
  echo "RUN model=$model"
  run_py "${args[@]}" 2>&1 | tee "$log"
done

run_py - "$OUT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted((out_dir / "summaries").glob("*_language_preference_stress_summary.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
if rows:
    merged = out_dir / "language_preference_stress_summary_all.csv"
    with merged.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged_summary={merged}")
else:
    print("no summaries found")
PY

echo "Language-preference stress diagnostic complete: $OUT_DIR"
