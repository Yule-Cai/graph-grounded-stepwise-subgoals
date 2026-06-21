#!/usr/bin/env python3
"""Write a compact Paper A experiment report from final rerun summaries."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _fmt_ci(row: dict[str, str], key: str) -> str:
    lo = row.get(f"{key}_ci95_low", "")
    hi = row.get(f"{key}_ci95_high", "")
    if not lo or not hi:
        return ""
    return f"[{_fmt(float(lo))}, {_fmt(float(hi))}]"


def _sort_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("scenario", ""),
        row.get("algo", ""),
        row.get("planner_mode", ""),
        row.get("model", ""),
    )


def _best_step_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    step_rows = [r for r in rows if r.get("planner_mode") == "llm_step"]
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in step_rows:
        key = (row.get("scenario", ""), row.get("algo", ""))
        if key not in best or _f(row, "success_rate") > _f(best[key], "success_rate"):
            best[key] = row
    return [best[k] for k in sorted(best)]


def _load_lmstudio_models(path: Path) -> str:
    if not path.exists():
        return "not recorded"
    text = path.read_text(errors="replace")
    start = text.find("{")
    if start < 0:
        return text.strip()[:500] or "not recorded"
    try:
        data = json.loads(text[start:])
    except json.JSONDecodeError:
        return text.strip()[:500]
    ids = [item.get("id", "") for item in data.get("data", []) if item.get("id")]
    return ", ".join(ids) if ids else "not recorded"


def _write_table(lines: list[str], rows: list[dict[str, str]]) -> None:
    if not rows:
        lines.append("No rows available yet.\n")
        return
    lines.append("| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |")
    lines.append("|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=_sort_key):
        lines.append(
            "| {scenario} | {algo} | {mode} | {model} | {episodes} | {success} | {ci} | {strict} | {parse} | {collision} | {timeout} | {sem:.3f} |".format(
                scenario=row.get("scenario", ""),
                algo=row.get("algo", ""),
                mode=row.get("planner_mode", ""),
                model=row.get("model", ""),
                episodes=row.get("episodes", ""),
                success=_fmt(_f(row, "success_rate")),
                ci=_fmt_ci(row, "success"),
                strict=_fmt(_f(row, "strict_valid_rate")),
                parse=_fmt(_f(row, "parse_ok_rate")),
                collision=_fmt(_f(row, "collision_rate")),
                timeout=_fmt(_f(row, "timeout_rate")),
                sem=_f(row, "mean_semantic_cost"),
            )
        )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--model-list", default="")
    parser.add_argument("--primary-model", default="")
    parser.add_argument("--zip-path", default="")
    args = parser.parse_args()

    rows = _read_csv(args.summary_csv)
    best_rows = _best_step_rows(rows)
    no_llm_rows = [r for r in rows if r.get("planner_mode") == "no_llm"]
    raw_rows = [r for r in rows if r.get("planner_mode") == "llm_raw"]
    step_rows = [r for r in rows if r.get("planner_mode") == "llm_step"]
    sac_rows = [r for r in rows if r.get("algo") == "sac"]

    lines: list[str] = []
    lines.append("# Paper A Final Experiment Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Run directory: `{args.run_dir}`")
    lines.append(f"Episodes per job requested: `{args.episodes}`")
    lines.append(f"Controller timesteps: `{args.timesteps}`")
    lines.append(f"Primary model: `{args.primary_model}`")
    lines.append(f"Model sweep: `{args.model_list}`")
    lines.append(f"LM Studio models visible at start: {_load_lmstudio_models(args.run_dir / 'lmstudio_models_at_start.json')}")
    if args.zip_path:
        lines.append(f"Packaged bundle: `{args.zip_path}`")
    lines.append("")

    lines.append("## Reviewer-Facing Experiment Plan")
    lines.append("")
    lines.append("- Main method: `llm_step`, where the local LLM chooses one legal next subgoal at a time from the structured topological map.")
    lines.append("- Failure-mode baseline: `llm_raw`, where the LLM directly proposes a full route without repair.")
    lines.append("- Classical baseline: `no_llm`, which removes language planning and uses the deterministic graph route.")
    lines.append("- Controller check: `sac` rows test whether the high-level route interface depends only on PPO.")
    lines.append("")

    lines.append("## Best Stepwise LLM Rows")
    lines.append("")
    _write_table(lines, best_rows)

    lines.append("## Stepwise LLM Model Sweep")
    lines.append("")
    _write_table(lines, step_rows)

    lines.append("## Raw Full-Route LLM Baseline")
    lines.append("")
    _write_table(lines, raw_rows)

    lines.append("## No-LLM Baseline")
    lines.append("")
    _write_table(lines, no_llm_rows)

    if sac_rows:
        lines.append("## Controller Ablation")
        lines.append("")
        _write_table(lines, sac_rows)

    lines.append("## Writing Notes")
    lines.append("")
    lines.append("- Use `llm_step` as the proposed method, not the old repair-heavy route generation pipeline.")
    lines.append("- Do not claim that the LLM universally beats graph search. The stronger claim is that planning granularity determines whether small local LLMs can produce executable semantic subgoal chains.")
    lines.append("- In the paper, emphasize `llm_raw` failures as evidence against one-shot full-route generation and `no_llm` as a strong non-language reference point.")
    lines.append("- If semantic-constraint success is weaker than long-horizon success, describe it as a limitation and use semantic cost / violation rate carefully.")
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {args.out_md}")


if __name__ == "__main__":
    main()
