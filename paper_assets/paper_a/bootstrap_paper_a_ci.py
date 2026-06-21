#!/usr/bin/env python3
"""Bootstrap confidence intervals for Paper A episode-level experiment CSVs."""

from __future__ import annotations

import argparse
import csv
import glob
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from summarize_revision_experiments import _float, _group_key


MetricFn = Callable[[list[dict[str, str]]], float]


def read_rows(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def rate(key: str) -> MetricFn:
    def fn(rows: list[dict[str, str]]) -> float:
        return mean([_float(row, key) for row in rows]) if rows else 0.0

    return fn


def numeric_mean(key: str) -> MetricFn:
    def fn(rows: list[dict[str, str]]) -> float:
        values = [_float(row, key) for row in rows if row.get(key, "") != ""]
        return mean(values) if values else 0.0

    return fn


METRICS: dict[str, MetricFn] = {
    "success_rate": rate("success"),
    "strict_valid_rate": rate("plan_valid"),
    "parse_ok_rate": rate("parse_ok"),
    "collision_rate": rate("collision"),
    "timeout_rate": rate("timeout"),
    "mean_steps": numeric_mean("steps"),
    "mean_semantic_cost": numeric_mean("semantic_cost"),
    "mean_route_distance": numeric_mean("route_distance"),
    "mean_route_turns": numeric_mean("route_turns"),
}


def bootstrap(group: list[dict[str, str]], metric: MetricFn, draws: int, rng: random.Random) -> tuple[float, float, float]:
    observed = metric(group)
    if not group:
        return observed, 0.0, 0.0
    values: list[float] = []
    n = len(group)
    for _ in range(draws):
        sample = [group[rng.randrange(n)] for _ in range(n)]
        values.append(metric(sample))
    return observed, percentile(values, 2.5), percentile(values, 97.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv-glob", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--group-by-map", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.episode_csv_glob)
    if not rows:
        raise SystemExit(f"No episode rows matched {args.episode_csv_glob}")

    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scenario, algo, mode, model = _group_key(row)
        if args.group_by_map:
            grouped[(scenario, algo, mode, model, row.get("map_id", ""))].append(row)
        else:
            grouped[(scenario, algo, mode, model)].append(row)

    rng = random.Random(args.seed)
    out_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        if args.group_by_map:
            scenario, algo, mode, model, map_id = key
        else:
            scenario, algo, mode, model = key
            map_id = "all"
        out: dict[str, Any] = {
            "scenario": scenario,
            "algo": algo,
            "planner_mode": mode,
            "model": model,
            "map_id": map_id,
            "episodes": len(group),
            "unique_cases": len({row.get("case_id", "") for row in group}),
            "unique_seeds": len({row.get("seed", "") for row in group}),
        }
        for metric_name, metric_fn in METRICS.items():
            value, low, high = bootstrap(group, metric_fn, args.bootstrap_samples, rng)
            out[metric_name] = value
            out[f"{metric_name}_boot_low"] = low
            out[f"{metric_name}_boot_high"] = high
        out_rows.append(out)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    for row in out_rows:
        print(
            f"{row['scenario']} {row['planner_mode']} {row['model']} map={row['map_id']} "
            f"n={row['episodes']} success={row['success_rate']:.3f} "
            f"boot95=[{row['success_rate_boot_low']:.3f},{row['success_rate_boot_high']:.3f}]"
        )


if __name__ == "__main__":
    main()
