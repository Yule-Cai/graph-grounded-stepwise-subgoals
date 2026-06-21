#!/usr/bin/env zsh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EPISODES="${1:-250}"
TIMESTEPS="${2:-5000000}"
STAMP="$(date +"%Y%m%d_%H%M%S")"

# Five basic preference classes x 50 base cases = 250 closed-loop episodes.
# The script reuses the stronger LLM--planner hybrid baselines:
#   - language_to_cost: LLM chooses a risk weight, Dijkstra plans.
#   - route_option_rank: graph planner generates route options, LLM ranks them.
OUT_DIR="${OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/preference_250_upgrade_${STAMP}}"
CASE_INPUT="${CASE_INPUT:-$WORKSPACE/paper_a_experiments_desktop/experiments/paper_a/cases/proactive_semantic_constraint_100.csv}"
BASE_LIMIT="${BASE_LIMIT:-50}"
PREFERENCE_SET="${PREFERENCE_SET:-basic}"
MODEL_LIST="${MODEL_LIST:-liquid/lfm2.5-1.2b nvidia/nemotron-3-nano-4b}"
BASELINE_MODES="${BASELINE_MODES:-no_llm graph_shortest weighted_scorer preference_scorer}"
LLM_BASELINE_MODES="${LLM_BASELINE_MODES:-language_to_cost route_option_rank}"
RUN_DETERMINISTIC_BASELINES="${RUN_DETERMINISTIC_BASELINES:-1}"
RUN_LLM_BASELINES="${RUN_LLM_BASELINES:-1}"
ROUTE_OPTION_RISK_WEIGHTS="${ROUTE_OPTION_RISK_WEIGHTS:-0,1,2.5,5,10,15}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MIN_COMPLETED_ROWS="${MIN_COMPLETED_ROWS:-$EPISODES}"

echo "Paper A preference 250-upgrade run"
echo "out_dir=$OUT_DIR"
echo "episodes=$EPISODES base_limit=$BASE_LIMIT preference_set=$PREFERENCE_SET"
echo "case_input=$CASE_INPUT"
echo "models=$MODEL_LIST"
echo "baselines=$BASELINE_MODES"
echo "llm_hybrid_modes=$LLM_BASELINE_MODES"
echo "run_deterministic=$RUN_DETERMINISTIC_BASELINES run_llm=$RUN_LLM_BASELINES"

OUT_DIR="$OUT_DIR" \
CASE_INPUT="$CASE_INPUT" \
BASE_LIMIT="$BASE_LIMIT" \
PREFERENCE_SET="$PREFERENCE_SET" \
MODEL_LIST="$MODEL_LIST" \
BASELINE_MODES="$BASELINE_MODES" \
LLM_BASELINE_MODES="$LLM_BASELINE_MODES" \
RUN_DETERMINISTIC_BASELINES="$RUN_DETERMINISTIC_BASELINES" \
RUN_LLM_BASELINES="$RUN_LLM_BASELINES" \
ROUTE_OPTION_RISK_WEIGHTS="$ROUTE_OPTION_RISK_WEIGHTS" \
SKIP_COMPLETED="$SKIP_COMPLETED" \
MIN_COMPLETED_ROWS="$MIN_COMPLETED_ROWS" \
zsh "$WORKSPACE/paper_assets/paper_a/run_language_cost_option_baselines.zsh" "$EPISODES" "$TIMESTEPS"

echo
echo "Done. Reviewer-facing outputs:"
echo "  $OUT_DIR/language_cost_option_summary_with_ci.csv"
echo "  $OUT_DIR/preference_by_type_summary.csv"
