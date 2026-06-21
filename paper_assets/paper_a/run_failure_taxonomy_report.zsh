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
SCRIPT="$WORKSPACE/paper_assets/paper_a/generate_failure_taxonomy_report.py"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/failure_taxonomy_${STAMP}}"
MAX_BARS="${MAX_BARS:-18}"

patterns=(
  "$WORKSPACE/paper_assets/paper_a/rerun_logs/final_aaai_stepwise_20260602_050932/episodes/*.csv"
  "$WORKSPACE/paper_assets/paper_a/rerun_logs/supplemental_reviewer_20260603/episodes/*.csv"
  "$WORKSPACE/paper_assets/paper_a/rerun_logs/supplemental_reviewer_20260603_ablation/episodes/*.csv"
  "$WORKSPACE/paper_assets/paper_a/rerun_logs/order_gate_innovation_20260616/episodes/*.csv"
  "$WORKSPACE/paper_assets/paper_a/rerun_logs/reviewer_gap_20260612/closed_loop_preference/episodes/*.csv"
  "$WORKSPACE/paper_assets/paper_a/rerun_logs/language_value_upgrade_*/**/episodes/*.csv"
  "$WORKSPACE/paper_assets/paper_a/rerun_logs/graph_perturbation_*/episodes/*.csv"
)

args=("$SCRIPT" --out-dir "$OUT_DIR" --max-bars "$MAX_BARS")
for pattern in "${patterns[@]}"; do
  args+=(--episode-csv-glob "$pattern")
done

echo "Generating failure taxonomy report"
echo "out_dir=$OUT_DIR"
run_py "${args[@]}"
