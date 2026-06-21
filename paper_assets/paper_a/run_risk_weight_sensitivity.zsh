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
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/risk_weight_sensitivity_${STAMP}}"
WEIGHTS=(${=WEIGHTS:-0 1 2.5 5 10})
CASE_FILE="${CASE_FILE:-$DESKTOP_ROOT/experiments/paper_a/cases/proactive_semantic_constraint_100.csv}"
MAP_ID="${MAP_ID:-reference_family_flat}"
MAP_SOURCE="${MAP_SOURCE:-gazebo_3d_projection}"
MAIN_ALGO="${MAIN_ALGO:-ppo}"
MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
BOOTSTRAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/bootstrap_paper_a_ci.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs"

csv_data_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local lines=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do lines=$((lines + 1)); done < "$path"
  if (( lines <= 1 )); then echo 0; else echo $((lines - 1)); fi
}

model_file_for_algo() {
  local algo="$1"
  case "$algo" in
    ppo) echo "models/PPO/ppo_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    sac) echo "models/SAC/sac_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    *) echo "ERROR_UNKNOWN_ALGO" ;;
  esac
}

safe_weight() {
  local value="$1"
  value="${value//./p}"
  echo "$value"
}

print_header "Paper A risk-weight sensitivity"
echo "out_dir=$OUT_DIR"
echo "case_file=$CASE_FILE weights=${WEIGHTS[*]} episodes=$EPISODES"

for weight in "${WEIGHTS[@]}"; do
  tag="$(safe_weight "$weight")"
  label="risk_weight_lambda_${tag}_${MAIN_ALGO}_no_llm"
  episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
  summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"
  log="$OUT_DIR/${label}_${EPISODES}ep.log"
  if [[ -f "$episode_csv" ]]; then
    rows="$(csv_data_rows "$episode_csv")"
    if (( rows >= EPISODES )); then
      echo "SKIP completed lambda=$weight rows=$rows"
      continue
    fi
  fi
  echo "RUN lambda=$weight"
  PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
    --algo "$MAIN_ALGO" \
    --model "$(model_file_for_algo "$MAIN_ALGO")" \
    --cases "$CASE_FILE" \
    --episodes "$EPISODES" \
    --seed 31 \
    --maps "$MAP_ID" \
    --map-source "$MAP_SOURCE" \
    --planner-mode no_llm \
    --lm-model no_llm \
    --semantic-cost-weight "$weight" \
    --repair-invalid-route \
    --run-label "$label" \
    --episode-csv "$episode_csv" \
    --summary-csv "$summary_csv" \
    2>&1 | tee "$log"
done

print_header "Summarize risk-weight sensitivity"
run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
  --out-csv "$OUT_DIR/risk_weight_sensitivity_summary_with_ci.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

run_py "$BOOTSTRAP_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --out-csv "$OUT_DIR/risk_weight_sensitivity_bootstrap_ci.csv" \
  --bootstrap-samples 2000 \
  2>&1 | tee "$OUT_DIR/stage_logs/bootstrap_ci.log"

run_py - "$OUT_DIR" <<'PY'
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted((out_dir / "episodes").glob("*.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
groups = defaultdict(list)
for row in rows:
    groups[row.get("run_label", "")].append(row)

def f(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0

def mean(group, key, skip_blank=False):
    vals = []
    for row in group:
        if skip_blank and row.get(key, "") == "":
            continue
        vals.append(f(row, key))
    return sum(vals) / len(vals) if vals else 0.0

def percentile(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = round((p / 100.0) * (len(vals) - 1))
    return vals[max(0, min(len(vals) - 1, idx))]

def boot(group, key, draws=2000, seed=20260612):
    rng = random.Random(seed + sum(ord(ch) for ch in key))
    n = len(group)
    obs = mean(group, key, skip_blank=key in {"route_distance", "route_turns", "semantic_cost"})
    vals = []
    for _ in range(draws):
        sample = [group[rng.randrange(n)] for _ in range(n)]
        vals.append(mean(sample, key, skip_blank=key in {"route_distance", "route_turns", "semantic_cost"}))
    return obs, percentile(vals, 2.5), percentile(vals, 97.5)

out_rows = []
for label, group in sorted(groups.items()):
    match = re.search(r"lambda_([^_]+)_", label)
    lam = match.group(1).replace("p", ".") if match else ""
    row = {"run_label": label, "lambda": lam, "episodes": len(group)}
    for key, out_key in [
        ("success", "success_rate"),
        ("plan_valid", "strict_valid_rate"),
        ("collision", "collision_rate"),
        ("timeout", "timeout_rate"),
        ("semantic_cost", "mean_semantic_cost"),
        ("route_distance", "mean_route_distance"),
        ("route_turns", "mean_route_turns"),
    ]:
        obs, lo, hi = boot(group, key)
        row[out_key] = f"{obs:.4f}"
        row[f"{out_key}_low"] = f"{lo:.4f}"
        row[f"{out_key}_high"] = f"{hi:.4f}"
    out_rows.append(row)

if out_rows:
    out_path = out_dir / "risk_weight_sensitivity_by_lambda.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"by_lambda_summary={out_path}")
PY

echo "Risk-weight sensitivity complete: $OUT_DIR"
