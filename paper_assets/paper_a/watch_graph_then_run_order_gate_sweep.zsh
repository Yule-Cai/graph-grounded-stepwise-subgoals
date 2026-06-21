#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EPISODES="${1:-100}"
TIMESTEPS="${2:-5000000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
POST_WAIT_SECONDS="${POST_WAIT_SECONDS:-180}"

CURRENT_GRAPH_OUT_DIR="${CURRENT_GRAPH_OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/graph_perturbation_llm_gate_20260618}"
NEXT_OUT_DIR="${NEXT_OUT_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/order_gate_threshold_sweep_20260618}"
FAILURE_OUT_DIR="${FAILURE_OUT_DIR:-$WORKSPACE/paper_assets/paper_a/failure_taxonomy_after_gate_sweep}"

MODEL_LIST="${MODEL_LIST:-liquid/lfm2.5-1.2b}"
MAP_LIST="${MAP_LIST:-reference_family_flat}"
SCENARIOS="${SCENARIOS:-long_horizon semantic_constraint}"
GATE_CONFIGS="${GATE_CONFIGS:-lenient:3:2:0.50 balanced:5:3:0.60 strict:5:4:0.80}"
RUN_REFERENCE_ABLATION="${RUN_REFERENCE_ABLATION:-1}"
RUN_GATE_SWEEP="${RUN_GATE_SWEEP:-1}"
RUN_FAILURE_TAXONOMY="${RUN_FAILURE_TAXONOMY:-1}"
UNLOAD_LMSTUDIO_AT_END="${UNLOAD_LMSTUDIO_AT_END:-1}"
AUTO_LOAD_LMSTUDIO_MODEL="${AUTO_LOAD_LMSTUDIO_MODEL:-1}"
LMS_GPU="${LMS_GPU:-max}"
GRAPH_EXPECTED="${GRAPH_EXPECTED:-24}"

active_graph_processes() {
  ps -axo pid=,command= \
    | grep -F "$CURRENT_GRAPH_OUT_DIR" \
    | grep -E 'run_graph_perturbation|run_map_conditioned_llm_planning' \
    | grep -v grep || true
}

csv_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local lines=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do
    lines=$((lines + 1))
  done < "$path"
  if (( lines <= 1 )); then echo 0; else echo $((lines - 1)); fi
}

progress_bar() {
  local done="$1"
  local total="$2"
  local width="${3:-30}"
  local pct=0 filled=0 empty=0
  if (( total > 0 )); then
    pct=$((100 * done / total))
    filled=$((width * done / total))
  fi
  if (( filled > width )); then filled="$width"; fi
  empty=$((width - filled))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3d%%' "$pct"
}

print_graph_status() {
  local episode_dir="$CURRENT_GRAPH_OUT_DIR/episodes"
  local summary_dir="$CURRENT_GRAPH_OUT_DIR/summaries"
  echo "---- graph perturbation LLM-gate status ----"
  if [[ -d "$episode_dir" ]]; then
    local full=0 partial=0 total=0 newest="" newest_rows=0 newest_mtime=0
    for csv in "$episode_dir"/*.csv(N); do
      total=$((total + 1))
      rows="$(csv_rows "$csv")"
      if (( rows >= EPISODES )); then
        full=$((full + 1))
      else
        partial=$((partial + 1))
        mtime="$(stat -f %m "$csv" 2>/dev/null || echo 0)"
        if (( mtime >= newest_mtime )); then
          newest_mtime="$mtime"
          newest="$csv"
          newest_rows="$rows"
        fi
      fi
    done
    local unit_done=$((full * EPISODES + newest_rows))
    local unit_total=$((GRAPH_EXPECTED * EPISODES))
    printf 'jobs:     %s  %d/%d complete, partial=%d, csv=%d, summaries=%s\n' \
      "$(progress_bar "$full" "$GRAPH_EXPECTED" 30)" "$full" "$GRAPH_EXPECTED" "$partial" "$total" \
      "$(find "$summary_dir" -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"
    printf 'episodes: %s  %d/%d episode-units\n' \
      "$(progress_bar "$unit_done" "$unit_total" 30)" "$unit_done" "$unit_total"
    if [[ -n "$newest" ]]; then
      echo "current:  ${newest:t} rows=$newest_rows/$EPISODES"
    fi
    for csv in "$episode_dir"/*.csv(N[-5,-1]); do
      printf "%4s rows  %s\n" "$(csv_rows "$csv")" "${csv:t}"
    done
  else
    echo "episode_dir missing: $episode_dir"
  fi
}

echo "============================================================"
echo "Paper A graph-to-gate-sweep watcher"
echo "current_graph_out_dir=$CURRENT_GRAPH_OUT_DIR"
echo "next_out_dir=$NEXT_OUT_DIR"
echo "episodes=$EPISODES timesteps=$TIMESTEPS"
echo "models=$MODEL_LIST maps=$MAP_LIST scenarios=$SCENARIOS"
echo "gate_configs=$GATE_CONFIGS"
echo "============================================================"

while true; do
  active="$(active_graph_processes)"
  if [[ -z "$active" ]]; then
    echo "No active graph perturbation process detected."
    print_graph_status
    break
  fi
  echo "Graph perturbation experiment is still running:"
  echo "$active"
  print_graph_status
  echo "Sleeping ${POLL_SECONDS}s before the next check..."
  sleep "$POLL_SECONDS"
done

if (( POST_WAIT_SECONDS > 0 )); then
  echo "Waiting ${POST_WAIT_SECONDS}s before starting the next LLM stage..."
  sleep "$POST_WAIT_SECONDS"
fi

if [[ "$AUTO_LOAD_LMSTUDIO_MODEL" == "1" ]] && command -v lms >/dev/null 2>&1; then
  model_tokens=(${=MODEL_LIST})
  first_model="${model_tokens[1]:-}"
  if [[ -n "$first_model" ]]; then
    echo "============================================================"
    echo "Preloading LM Studio model: $first_model"
    echo "============================================================"
    lms load "$first_model" --gpu "$LMS_GPU" -y || true
  fi
fi

echo "============================================================"
echo "Stage B: order-gate threshold sweep"
echo "============================================================"
OUT_DIR="$NEXT_OUT_DIR" \
MODEL_LIST="$MODEL_LIST" \
MAP_LIST="$MAP_LIST" \
SCENARIOS="$SCENARIOS" \
GATE_CONFIGS="$GATE_CONFIGS" \
RUN_REFERENCE_ABLATION="$RUN_REFERENCE_ABLATION" \
RUN_GATE_SWEEP="$RUN_GATE_SWEEP" \
SKIP_COMPLETED=1 \
zsh "$WORKSPACE/paper_assets/paper_a/run_order_gate_threshold_sweep.zsh" "$EPISODES" "$TIMESTEPS"

if [[ "$RUN_FAILURE_TAXONOMY" == "1" ]]; then
  echo "============================================================"
  echo "Stage C: regenerate failure taxonomy report"
  echo "============================================================"
  OUT_DIR="$FAILURE_OUT_DIR" \
  MAX_BARS=28 \
  zsh "$WORKSPACE/paper_assets/paper_a/run_failure_taxonomy_report.zsh"
fi

if [[ "$UNLOAD_LMSTUDIO_AT_END" == "1" ]] && command -v lms >/dev/null 2>&1; then
  echo "============================================================"
  echo "Final cleanup: unloading LM Studio models"
  echo "============================================================"
  lms unload --all || true
fi

echo "============================================================"
echo "Graph-to-gate-sweep watcher complete"
echo "next_out_dir=$NEXT_OUT_DIR"
echo "failure_out_dir=$FAILURE_OUT_DIR"
echo "============================================================"
