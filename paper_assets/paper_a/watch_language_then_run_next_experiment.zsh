#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
POLL_SECONDS="${POLL_SECONDS:-120}"

LANGUAGE_OUT_ROOT="${LANGUAGE_OUT_ROOT:-$WORKSPACE/paper_assets/paper_a/rerun_logs/language_value_upgrade_20260617}"
NEXT_OUT_DIR="${NEXT_OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/graph_perturbation_llm_gate_20260618}"
FAILURE_OUT_DIR="${FAILURE_OUT_DIR:-$WORKSPACE/paper_assets/paper_a/failure_taxonomy_after_next}"

RUN_NEXT_GRAPH_LLM="${RUN_NEXT_GRAPH_LLM:-1}"
RUN_FAILURE_TAXONOMY="${RUN_FAILURE_TAXONOMY:-1}"
UNLOAD_LMSTUDIO_AT_END="${UNLOAD_LMSTUDIO_AT_END:-1}"

NEXT_MODEL_LIST="${NEXT_MODEL_LIST:-liquid/lfm2.5-1.2b}"
NEXT_MAP_LIST="${NEXT_MAP_LIST:-reference_family_flat reference_villa_ground studio_apartment}"
NEXT_SCENARIOS="${NEXT_SCENARIOS:-long_horizon semantic_constraint}"
NEXT_PERTURBATIONS="${NEXT_PERTURBATIONS:-clean edge_drop risk_noise combined}"
NEXT_LLM_MODES="${NEXT_LLM_MODES:-llm_step_consistency_gate}"

active_language_processes() {
  ps -axo pid=,command= \
    | grep -F "$LANGUAGE_OUT_ROOT" \
    | grep -E 'run_language_value_upgrade|run_closed_loop_preference|run_map_conditioned_llm_planning' \
    | grep -v grep || true
}

csv_data_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local lines=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do
    lines=$((lines + 1))
  done < "$path"
  if (( lines <= 1 )); then echo 0; else echo $((lines - 1)); fi
}

print_closed_loop_status() {
  local episode_dir="$LANGUAGE_OUT_ROOT/closed_loop_preference_order_robust/episodes"
  local summary_dir="$LANGUAGE_OUT_ROOT/closed_loop_preference_order_robust/summaries"
  echo "---- closed-loop preference status ----"
  if [[ -d "$episode_dir" ]]; then
    for csv in "$episode_dir"/*.csv(N); do
      printf "%4s rows  %s\n" "$(csv_data_rows "$csv")" "${csv:t}"
    done
  else
    echo "episode_dir missing: $episode_dir"
  fi
  echo "summaries=$(find "$summary_dir" -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"
}

echo "============================================================"
echo "Paper A watch-and-chain runner"
echo "language_out_root=$LANGUAGE_OUT_ROOT"
echo "next_out_dir=$NEXT_OUT_DIR"
echo "episodes=$EPISODES timesteps=$TIMESTEPS poll_seconds=$POLL_SECONDS"
echo "run_next_graph_llm=$RUN_NEXT_GRAPH_LLM run_failure_taxonomy=$RUN_FAILURE_TAXONOMY"
echo "============================================================"

print_closed_loop_status

while true; do
  active="$(active_language_processes)"
  if [[ -z "$active" ]]; then
    echo "No active language-value process detected."
    break
  fi
  echo "Language-value experiment is still running:"
  echo "$active"
  print_closed_loop_status
  echo "Sleeping ${POLL_SECONDS}s before the next check..."
  sleep "$POLL_SECONDS"
done

echo "============================================================"
echo "Stage A complete: language-value runner has finished"
echo "============================================================"

if [[ "$RUN_NEXT_GRAPH_LLM" == "1" ]]; then
  echo "============================================================"
  echo "Stage B: graph perturbation LLM gate experiment"
  echo "models=$NEXT_MODEL_LIST"
  echo "maps=$NEXT_MAP_LIST"
  echo "scenarios=$NEXT_SCENARIOS"
  echo "perturbations=$NEXT_PERTURBATIONS"
  echo "llm_modes=$NEXT_LLM_MODES"
  echo "============================================================"
  OUT_DIR="$NEXT_OUT_DIR" \
  MODEL_LIST="$NEXT_MODEL_LIST" \
  MAP_LIST="$NEXT_MAP_LIST" \
  SCENARIOS="$NEXT_SCENARIOS" \
  PERTURBATIONS="$NEXT_PERTURBATIONS" \
  LLM_MODES="$NEXT_LLM_MODES" \
  RUN_BASELINES=0 \
  RUN_LLM=1 \
  SKIP_COMPLETED=1 \
  zsh "$WORKSPACE/paper_assets/paper_a/run_graph_perturbation_stress_experiments.zsh" "$EPISODES" "$TIMESTEPS"
else
  echo "Skipping graph perturbation LLM stage because RUN_NEXT_GRAPH_LLM=0."
fi

if [[ "$RUN_FAILURE_TAXONOMY" == "1" ]]; then
  echo "============================================================"
  echo "Stage C: regenerate failure taxonomy report"
  echo "failure_out_dir=$FAILURE_OUT_DIR"
  echo "============================================================"
  OUT_DIR="$FAILURE_OUT_DIR" \
  MAX_BARS=24 \
  zsh "$WORKSPACE/paper_assets/paper_a/run_failure_taxonomy_report.zsh"
else
  echo "Skipping failure taxonomy stage because RUN_FAILURE_TAXONOMY=0."
fi

if [[ "$UNLOAD_LMSTUDIO_AT_END" == "1" ]] && command -v lms >/dev/null 2>&1; then
  echo "============================================================"
  echo "Final cleanup: unloading LM Studio models"
  echo "============================================================"
  lms unload --all || true
fi

echo "============================================================"
echo "Watch-and-chain runner complete"
echo "next_out_dir=$NEXT_OUT_DIR"
echo "failure_out_dir=$FAILURE_OUT_DIR"
echo "============================================================"
