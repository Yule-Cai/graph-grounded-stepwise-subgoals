#!/usr/bin/env zsh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EPISODES="${EPISODES:-100}"
REFRESH_SECONDS="${REFRESH_SECONDS:-60}"
WATCH="${WATCH:-0}"

GRAPH_DIR="${GRAPH_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/graph_perturbation_llm_gate_20260618}"
GATE_DIR="${GATE_DIR:-$WORKSPACE/paper_assets/paper_a/rerun_logs/order_gate_threshold_sweep_20260618}"
FAILURE_DIR="${FAILURE_DIR:-$WORKSPACE/paper_assets/paper_a/failure_taxonomy_after_gate_sweep}"

GRAPH_EXPECTED="${GRAPH_EXPECTED:-24}"
GATE_EXPECTED="${GATE_EXPECTED:-14}"

csv_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo 0; return 0; fi
  local lines=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do
    lines=$((lines + 1))
  done < "$path"
  if (( lines <= 1 )); then echo 0; else echo $((lines - 1)); fi
}

bar() {
  local done="$1"
  local total="$2"
  local width="${3:-34}"
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

status_for_dir() {
  local title="$1"
  local dir="$2"
  local expected="$3"
  local episode_dir="$dir/episodes"
  local summary_dir="$dir/summaries"
  local full=0 partial=0 current_rows=0 total_csv=0
  local newest="" newest_mtime=0
  if [[ -d "$episode_dir" ]]; then
    for csv in "$episode_dir"/*.csv(N); do
      total_csv=$((total_csv + 1))
      rows="$(csv_rows "$csv")"
      if (( rows >= EPISODES )); then
        full=$((full + 1))
      else
        partial=$((partial + 1))
        mtime="$(stat -f %m "$csv" 2>/dev/null || echo 0)"
        if (( mtime >= newest_mtime )); then
          newest_mtime="$mtime"
          newest="$csv"
          current_rows="$rows"
        fi
      fi
    done
  fi
  local unit_done=$((full * EPISODES + current_rows))
  local unit_total=$((expected * EPISODES))
  echo "$title"
  printf '  jobs: %s  %d/%d complete, partial=%d, csv=%d, summaries=%s\n' \
    "$(bar "$full" "$expected" 30)" \
    "$full" "$expected" "$partial" "$total_csv" \
    "$(find "$summary_dir" -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"
  printf '  episodes: %s  %d/%d episode-units\n' \
    "$(bar "$unit_done" "$unit_total" 30)" \
    "$unit_done" "$unit_total"
  if [[ -n "$newest" ]]; then
    echo "  current: ${newest:t} rows=$current_rows/$EPISODES"
  fi
}

print_once() {
  echo "============================================================"
  echo "Paper A Progress Monitor  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  status_for_dir "Graph perturbation LLM-gate" "$GRAPH_DIR" "$GRAPH_EXPECTED"
  status_for_dir "Order-gate threshold sweep" "$GATE_DIR" "$GATE_EXPECTED"
  echo "Failure taxonomy"
  if [[ -d "$FAILURE_DIR" ]]; then
    find "$FAILURE_DIR" -maxdepth 1 -type f -print | sed 's#^#  #'
  else
    echo "  pending"
  fi
  echo "Active processes"
  ps -axo pid=,etime=,pcpu=,pmem=,command= \
    | grep -E 'watch_graph_then|run_graph_perturbation|run_order_gate_threshold|run_map_conditioned_llm_planning' \
    | grep -v grep \
    | sed 's#^#  #' || echo "  none"
}

if [[ "$WATCH" == "1" ]]; then
  while true; do
    clear
    print_once
    sleep "$REFRESH_SECONDS"
  done
else
  print_once
fi
