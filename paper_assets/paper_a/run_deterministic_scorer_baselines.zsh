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
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/deterministic_scorer_baselines_${STAMP}}"
SCENARIOS=(${=SCENARIOS:-long_horizon semantic_constraint})
MODES=(${=MODES:-first_candidate weighted_scorer preference_scorer})
MAP_ID="${MAP_ID:-reference_family_flat}"
MAP_SOURCE="${MAP_SOURCE:-gazebo_3d_projection}"
MAIN_ALGO="${MAIN_ALGO:-ppo}"
MAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/run_map_conditioned_llm_planning.py"
SUMMARY_SCRIPT="$WORKSPACE/paper_assets/paper_a/summarize_revision_experiments.py"
BOOTSTRAP_SCRIPT="$WORKSPACE/paper_assets/paper_a/bootstrap_paper_a_ci.py"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/stage_logs"

model_file_for_algo() {
  local algo="$1"
  case "$algo" in
    ppo) echo "models/PPO/ppo_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    sac) echo "models/SAC/sac_reference_family_flat_goalfirst_${TIMESTEPS}_from_scratch.zip" ;;
    *) echo "ERROR_UNKNOWN_ALGO" ;;
  esac
}

csv_data_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local lines=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do lines=$((lines + 1)); done < "$path"
  if (( lines <= 1 )); then echo 0; else echo $((lines - 1)); fi
}

print_header "Paper A deterministic scorer baselines"
echo "out_dir=$OUT_DIR"
echo "episodes=$EPISODES modes=${MODES[*]} scenarios=${SCENARIOS[*]}"

for scenario in "${SCENARIOS[@]}"; do
  case_file="$DESKTOP_ROOT/experiments/paper_a/cases/proactive_${scenario}_100.csv"
  for mode in "${MODES[@]}"; do
    label="${scenario}_${MAIN_ALGO}_${mode}_no_llm"
    episode_csv="$OUT_DIR/episodes/${label}_${EPISODES}ep.csv"
    summary_csv="$OUT_DIR/summaries/${label}_${EPISODES}ep.csv"
    log="$OUT_DIR/${label}_${EPISODES}ep.log"
    rows="$(csv_data_rows "$episode_csv")"
    if (( rows >= EPISODES )); then
      echo "SKIP completed scenario=$scenario mode=$mode rows=$rows"
      continue
    fi
    echo "RUN scenario=$scenario mode=$mode"
    PYTHONUNBUFFERED=1 run_py "$MAP_SCRIPT" \
      --algo "$MAIN_ALGO" \
      --model "$(model_file_for_algo "$MAIN_ALGO")" \
      --cases "$case_file" \
      --episodes "$EPISODES" \
      --seed 31 \
      --maps "$MAP_ID" \
      --map-source "$MAP_SOURCE" \
      --planner-mode "$mode" \
      --lm-model no_llm \
      --repair-invalid-route \
      --run-label "$label" \
      --episode-csv "$episode_csv" \
      --summary-csv "$summary_csv" \
      2>&1 | tee "$log"
  done
done

print_header "Summarize deterministic scorer baselines"
run_py "$SUMMARY_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --run-summary-csv-glob "$OUT_DIR/summaries/*.csv" \
  --out-csv "$OUT_DIR/deterministic_scorer_summary_with_ci.csv" \
  2>&1 | tee "$OUT_DIR/stage_logs/final_summary.log"

run_py "$BOOTSTRAP_SCRIPT" \
  --episode-csv-glob "$OUT_DIR/episodes/*.csv" \
  --out-csv "$OUT_DIR/deterministic_scorer_bootstrap_ci.csv" \
  --bootstrap-samples 2000 \
  2>&1 | tee "$OUT_DIR/stage_logs/bootstrap_ci.log"

echo "Deterministic scorer baselines complete: $OUT_DIR"
