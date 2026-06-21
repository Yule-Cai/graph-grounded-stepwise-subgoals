#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$WORKSPACE/paper_assets/paper_a/run_route_preference_alignment.py"

LIMIT="${1:-20}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/nl_preference_${STAMP}}"
CASES="${CASES:-$WORKSPACE/paper_assets/paper_a/raw/preference_alignment_cases.csv}"
MODEL_LIST="${MODEL_LIST:-liquid/lfm2.5-1.2b nvidia/nemotron-3-nano-4b}"
LM_STUDIO_URL="${LM_STUDIO_URL:-http://127.0.0.1:1234}"
TIMEOUT_S="${TIMEOUT_S:-45}"
MAX_TOKENS="${MAX_TOKENS:-512}"
VARIANTS_PER_PREFERENCE="${VARIANTS_PER_PREFERENCE:-2}"
OPTION_ORDER="${OPTION_ORDER:-shuffled}"
WRITE_PROMPTS_ONLY="${WRITE_PROMPTS_ONLY:-0}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  echo "ERROR: neither python3 nor python was found in PATH=$PATH" >&2
  exit 127
fi

mkdir -p "$OUT_DIR/results" "$OUT_DIR/summaries"

models=(${=MODEL_LIST})
total=${#models[@]}
done_count=0
start_epoch=$(date +%s)

progress() {
  local label="$1"
  local now elapsed eta pct filled empty bar
  now=$(date +%s)
  elapsed=$((now - start_epoch))
  if (( done_count > 0 )); then
    eta=$((elapsed * (total - done_count) / done_count))
  else
    eta=0
  fi
  if (( total > 0 )); then
    pct=$((100 * done_count / total))
    filled=$((32 * done_count / total))
  else
    pct=100
    filled=32
  fi
  empty=$((32 - filled))
  bar=""
  for ((i = 0; i < filled; i++)); do
    bar+="#"
  done
  for ((i = 0; i < empty; i++)); do
    bar+="-"
  done
  printf "[%s] %3d%% %s done=%d/%d elapsed=%02dm%02ds eta=%02dm%02ds\n" \
    "$bar" "$pct" "$label" "$done_count" "$total" \
    $((elapsed / 60)) $((elapsed % 60)) $((eta / 60)) $((eta % 60))
}

echo "Paper A natural-language preference diagnostic"
echo "out_dir=$OUT_DIR"
echo "cases=$CASES limit=$LIMIT variants_per_preference=$VARIANTS_PER_PREFERENCE option_order=$OPTION_ORDER"
echo "timeout_s=$TIMEOUT_S max_tokens=$MAX_TOKENS"
echo "models=$MODEL_LIST lm_url=$LM_STUDIO_URL"
echo "write_prompts_only=$WRITE_PROMPTS_ONLY"

for model in "${models[@]}"; do
  safe_model="${model//\//_}"
  result_csv="$OUT_DIR/results/${safe_model}_nl_preference.csv"
  summary_csv="$OUT_DIR/summaries/${safe_model}_nl_preference_summary.csv"

  progress "starting: $model"
  cmd=(
    "$PYTHON_BIN" "$RUNNER"
    --cases "$CASES"
    --output "$result_csv"
    --summary-output "$summary_csv"
    --limit "$LIMIT"
    --lm-studio-url "$LM_STUDIO_URL"
    --model "$model"
    --timeout-s "$TIMEOUT_S"
    --max-tokens "$MAX_TOKENS"
    --variants-per-preference "$VARIANTS_PER_PREFERENCE"
    --option-order "$OPTION_ORDER"
  )
  if [[ "$WRITE_PROMPTS_ONLY" == "1" ]]; then
    cmd+=(--write-prompts-only)
  fi
  "${cmd[@]}"
  status_code=$?
  if [[ $status_code -ne 0 ]]; then
    echo "ERROR: model $model failed with status $status_code" >&2
    exit $status_code
  fi
  done_count=$((done_count + 1))
  progress "finished: $model"
done

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary_files = sorted((out_dir / "summaries").glob("*_nl_preference_summary.csv"))
rows = []
for path in summary_files:
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
if rows:
    merged = out_dir / "nl_preference_summary_all.csv"
    with merged.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged_summary={merged}")
PY

echo "Natural-language preference diagnostic complete: $OUT_DIR"
