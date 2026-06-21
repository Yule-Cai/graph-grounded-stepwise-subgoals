#!/usr/bin/env python3
"""Summarize closed-loop preference runs by preference type.

The evaluator keeps the generated preference id in the case_id prefix, e.g.
``lowest_risk_semantic_constraint_012``.  This script turns the per-episode
CSVs into reviewer-facing per-preference metrics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


PREFERENCE_IDS = [
    "conflicting_kitchen_distance",
    "compositional_quiet_dry",
    "multi_risk_priority",
    "dynamic_update_battery",
    "disturbance_sensitive",
    "high_clearance",
    "lowest_risk",
    "fewest_turns",
    "balanced",
    "shortest",
]


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _pref_id(case_id: str) -> str:
    for pref in PREFERENCE_IDS:
        if case_id.startswith(pref + "_"):
            return pref
    return "unknown"


def _method_from_path(path: Path, row: dict[str, str]) -> tuple[str, str]:
    mode = row.get("planner_mode") or "unknown"
    run_label = row.get("run_label") or path.stem
    model = "no_llm"
    marker = f"_{mode}_"
    if marker in run_label:
        model = run_label.split(marker, 1)[1]
        model = model.rsplit("_", 1)[0] if model.endswith("ep") else model
    if mode in {"no_llm", "graph_shortest", "weighted_scorer", "preference_scorer", "first_candidate"}:
        model = "no_llm"
    model = model.replace("_", "/") if "/" not in model and model != "no_llm" else model
    return mode, model


def _rate(rows: list[dict[str, str]], key: str) -> float:
    return mean(_float(row, key) for row in rows) if rows else 0.0


def _mean_nonempty(rows: list[dict[str, str]], key: str) -> float:
    vals = [_float(row, key, math.nan) for row in rows if row.get(key, "") not in {"", "nan", "None"}]
    vals = [v for v in vals if not math.isnan(v)]
    return mean(vals) if vals else 0.0


def summarize(paths: list[Path]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                mode, model = _method_from_path(path, row)
                pref = row.get("preference_id") or _pref_id(row.get("case_id", ""))
                groups[(mode, model, pref)].append(row)

    out: list[dict[str, object]] = []
    for (mode, model, pref), rows in sorted(groups.items()):
        out.append(
            {
                "planner_mode": mode,
                "model": model,
                "preference_id": pref,
                "episodes": len(rows),
                "success_rate": round(_rate(rows, "success"), 4),
                "strict_valid_rate": round(_rate(rows, "plan_valid"), 4),
                "parse_ok_rate": round(_rate(rows, "parse_ok"), 4),
                "collision_rate": round(_rate(rows, "collision"), 4),
                "timeout_rate": round(_rate(rows, "timeout"), 4),
                "mean_steps": round(_mean_nonempty(rows, "steps"), 3),
                "mean_route_distance": round(_mean_nonempty(rows, "route_distance"), 3),
                "mean_route_turns": round(_mean_nonempty(rows, "route_turns"), 3),
                "mean_semantic_cost": round(_mean_nonempty(rows, "semantic_cost"), 4),
                "mean_route_score": round(_mean_nonempty(rows, "route_score"), 3),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv-glob", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.episode_csv_glob))]
    if not paths:
        raise SystemExit(f"no episode CSVs matched {args.episode_csv_glob!r}")
    rows = summarize(paths)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} preference-by-type rows to {args.out_csv}")


if __name__ == "__main__":
    main()
