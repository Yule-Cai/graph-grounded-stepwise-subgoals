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
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/closed_loop_preference_${STAMP}}"
BASE_LIMIT="${BASE_LIMIT:-20}"
MAP_ID="${MAP_ID:-reference_family_flat}"
MAP_SOURCE="${MAP_SOURCE:-gazebo_3d_projection}"
MAIN_ALGO="${MAIN_ALGO:-ppo}"
LM_URL="${LM_URL:-http://127.0.0.1:1234}"
MODEL_LIST=(${=MODEL_LIST:-nvidia/nemotron-3-nano-4b google/gemma-4-e4b})
BASELINE_MODES=(${=BASELINE_MODES:-no_llm graph_shortest first_candidate weighted_scorer preference_scorer})
LLM_MODES=(${=LLM_MODES:-llm_step_order_ensemble llm_step_consistency_gate})
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_LLM_STEP="${RUN_LLM_STEP:-1}"
PREFERENCE_SET="${PREFERENCE_SET:-basic}"
ORDER_GATE_VARIANTS="${ORDER_GATE_VARIANTS:-5}"
ORDER_GATE_MIN_VOTES="${ORDER_GATE_MIN_VOTES:-3}"
ORDER_GATE_MIN_CONSISTENCY="${ORDER_GATE_MIN_CONSISTENCY:-0.60}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-$EPISODES}"

MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
CASE_SCRIPT="$WORKSPACE/paper_assets/paper_a/generate_closed_loop_preference_cases.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"
CASE_FILE="$OUT_DIR/cases/closed_loop_preference_${BASE_LIMIT}base_${EPISODES}ep.csv"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs" "$OUT_DIR/cases"

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

run_condition() {
  local mode="$1"
  local lm_model="$2"
  local model_tag="$(sanitize "$lm_model")"
  local model_file="$(model_file_for_algo "$MAIN_ALGO")"
  local label="closed_loop_preference_${MAIN_ALGO}_${mode}_${model_tag}"
  local episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
  local summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"
  local log="$OUT_DIR/${label}_${EPISODES}ep.log"
  if [[ "$SKIP_COMPLETED" == "1" ]]; then
    local rows="$(csv_data_rows "$episode_csv")"
    if (( rows >= MIN_COMPLETED_ROWS )); then
      echo "SKIP completed mode=$mode model=$lm_model rows=$rows"
      return 0
    fi
  fi
  echo "RUN mode=$mode model=$lm_model cases=$CASE_FILE"
  PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
    --algo "$MAIN_ALGO" \
    --model "$model_file" \
    --cases "$CASE_FILE" \
    --episodes "$EPISODES" \
    --seed 31 \
    --maps "$MAP_ID" \
    --map-source "$MAP_SOURCE" \
    --lm-studio-url "$LM_URL" \
    --lm-model "$lm_model" \
    --planner-mode "$mode" \
    --order-gate-variants "$ORDER_GATE_VARIANTS" \
    --order-gate-min-votes "$ORDER_GATE_MIN_VOTES" \
    --order-gate-min-consistency "$ORDER_GATE_MIN_CONSISTENCY" \
    --repair-invalid-route \
    --run-label "$label" \
    --episode-csv "$episode_csv" \
    --summary-csv "$summary_csv" \
    2>&1 | tee "$log"
}

print_header "Paper A closed-loop natural-language preference experiments"
echo "out_dir=$OUT_DIR"
echo "episodes=$EPISODES base_limit=$BASE_LIMIT preference_set=$PREFERENCE_SET map=$MAP_ID models=${MODEL_LIST[*]}"
echo "baseline_modes=${BASELINE_MODES[*]} llm_modes=${LLM_MODES[*]} order_gate_variants=$ORDER_GATE_VARIANTS"

run_py "$CASE_SCRIPT" \
  --input "$WORKSPACE/paper_assets/paper_a/raw/preference_alignment_cases.csv" \
  --output "$CASE_FILE" \
  --base-limit "$BASE_LIMIT" \
  --map-id "$MAP_ID" \
  --preference-set "$PREFERENCE_SET" \
  2>&1 | tee "$OUT_DIR/stage_logs/generate_cases.log"

if [[ "$RUN_BASELINES" == "1" ]]; then
  print_header "Stage 1: deterministic baselines"
  for mode in "${BASELINE_MODES[@]}"; do
    run_condition "$mode" "no_llm"
  done
fi

if [[ "$RUN_LLM_STEP" == "1" ]]; then
  print_header "Stage 2: LLM closed-loop preference"
  for model in "${MODEL_LIST[@]}"; do
    for mode in "${LLM_MODES[@]}"; do
      run_condition "$mode" "$model"
    done
  done
fi

print_header "Stage 3: summarize"
run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
  --out-csv "$OUT_DIR/closed_loop_preference_summary_with_ci.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

run_py - "$OUT_DIR" <<'PY'
import csv
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
    case_id = row.get("case_id", "")
    pref = case_id.split("_semantic_constraint_", 1)[0] if "_semantic_constraint_" in case_id else case_id.split("_", 1)[0]
    groups[(row.get("run_label", ""), row.get("planner_mode", ""), pref)].append(row)
out_rows = []
for (label, mode, pref), group in sorted(groups.items()):
    n = len(group)
    if not n:
        continue
    def mean(key):
        vals = [float(r[key]) for r in group if r.get(key) not in ("", None)]
        return sum(vals) / len(vals) if vals else 0.0
    out_rows.append({
        "run_label": label,
        "planner_mode": mode,
        "preference_id": pref,
        "episodes": n,
        "success_rate": f"{mean('success'):.4f}",
        "strict_valid_rate": f"{mean('plan_valid'):.4f}",
        "parse_ok_rate": f"{mean('parse_ok'):.4f}",
        "collision_rate": f"{mean('collision'):.4f}",
        "timeout_rate": f"{mean('timeout'):.4f}",
        "mean_route_distance": f"{mean('route_distance'):.4f}",
        "mean_route_turns": f"{mean('route_turns'):.4f}",
        "mean_semantic_cost": f"{mean('semantic_cost'):.4f}",
    })
if out_rows:
    out_path = out_dir / "closed_loop_preference_by_preference.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"by_preference_summary={out_path}")
PY

echo "Closed-loop preference experiments complete: $OUT_DIR"
