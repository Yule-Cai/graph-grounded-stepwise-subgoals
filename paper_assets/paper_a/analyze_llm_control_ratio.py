#!/usr/bin/env python3
"""Summarize how often the gated stepwise LLM actually controls a decision.

The consistency-gate planner can either accept the LLM-voted next node or fall
back to a deterministic graph scorer. This script reports the decision-level
LLM control ratio from episode CSVs, so the paper can distinguish "LLM makes
most subgoal choices" from "the gate mostly protects by falling back."
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["_source_csv"] = path
                    rows.append(row)
    return rows


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("run_label", "")),
        str(row.get("planner_mode", "")),
        str(row.get("lm_model", "")),
        str(row.get("scenario", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv-glob", action="append", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    rows = _read_rows(args.episode_csv_glob)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "order_gate_steps" not in row:
            continue
        if _float(row, "order_gate_steps") <= 0:
            continue
        groups[_group_key(row)].append(row)

    out_rows: list[dict[str, Any]] = []
    for (label, mode, model, scenario), group in sorted(groups.items()):
        episodes = len(group)
        steps = sum(_float(row, "order_gate_steps") for row in group)
        accepts = sum(_float(row, "order_gate_accepts") for row in group)
        fallbacks = sum(_float(row, "order_gate_fallbacks") for row in group)
        out_rows.append(
            {
                "run_label": label,
                "planner_mode": mode,
                "lm_model": model,
                "scenario": scenario,
                "episodes": episodes,
                "decision_steps": int(steps),
                "llm_accepted_decisions": int(accepts),
                "fallback_decisions": int(fallbacks),
                "llm_control_ratio": f"{accepts / max(steps, 1):.4f}",
                "fallback_ratio": f"{fallbacks / max(steps, 1):.4f}",
                "episode_any_llm_accept_rate": f"{sum(1 for row in group if _float(row, 'order_gate_accepts') > 0) / max(episodes, 1):.4f}",
                "episode_any_fallback_rate": f"{sum(1 for row in group if _float(row, 'order_gate_fallbacks') > 0) / max(episodes, 1):.4f}",
                "mean_success_rate": f"{sum(_float(row, 'success') for row in group) / max(episodes, 1):.4f}",
                "mean_semantic_cost": f"{sum(_float(row, 'semantic_cost') for row in group) / max(episodes, 1):.4f}",
                "mean_route_distance": f"{sum(_float(row, 'route_distance') for row in group) / max(episodes, 1):.4f}",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "run_label",
        "planner_mode",
        "lm_model",
        "scenario",
        "episodes",
        "decision_steps",
        "llm_accepted_decisions",
        "fallback_decisions",
        "llm_control_ratio",
        "fallback_ratio",
        "episode_any_llm_accept_rate",
        "episode_any_fallback_rate",
        "mean_success_rate",
        "mean_semantic_cost",
        "mean_route_distance",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# LLM Control Ratio",
            "",
            "Decision-level ratio for consistency-gated stepwise planning.",
            "",
            "| run | scenario | episodes | LLM control | fallback | success | semantic cost |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in out_rows:
            lines.append(
                "| {run} | {scenario} | {episodes} | {control} | {fallback} | {success} | {semantic} |".format(
                    run=row["run_label"],
                    scenario=row["scenario"],
                    episodes=row["episodes"],
                    control=row["llm_control_ratio"],
                    fallback=row["fallback_ratio"],
                    success=row["mean_success_rate"],
                    semantic=row["mean_semantic_cost"],
                )
            )
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"rows={len(out_rows)} out_csv={args.out_csv}")
    if args.out_md:
        print(f"out_md={args.out_md}")


if __name__ == "__main__":
    main()
