#!/usr/bin/env python3
"""Summarize Paper A revision experiment CSVs with simple confidence intervals."""

from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_TAG_ALIASES = {
    "google_gemma-4-12b": "google/gemma-4-12b",
    "google_gemma-4-e4b": "google/gemma-4-e4b",
    "google_gemma-3-1b": "google/gemma-3-1b",
    "qwen_qwen3-1.7b": "qwen/qwen3-1.7b",
    "qwenpaw-flash-9b": "qwenpaw-flash-9b",
    "liquid_lfm2.5-1.2b": "liquid/lfm2.5-1.2b",
    "nvidia_nemotron-3-nano-4b": "nvidia/nemotron-3-nano-4b",
    "openai_gpt-oss-120b_free": "openai/gpt-oss-120b:free",
}


def _restore_model_tag(tag: str) -> str:
    if not tag:
        return "unknown"
    if tag in MODEL_TAG_ALIASES:
        return MODEL_TAG_ALIASES[tag]
    if tag.startswith("google_"):
        return "google/" + tag[len("google_") :]
    if tag.startswith("qwen_"):
        return "qwen/" + tag[len("qwen_") :]
    if tag.startswith("liquid_"):
        return "liquid/" + tag[len("liquid_") :]
    if tag.startswith("nvidia_"):
        return "nvidia/" + tag[len("nvidia_") :]
    return tag


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _read_rows(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows


def _parse_run_label(label: str, row: dict[str, str]) -> tuple[str, str, str, str]:
    for prefix in ("language_cost_option_", "preference_"):
        if not label.startswith(prefix):
            continue
        rest = label[len(prefix) :]
        for algo in ("ppo", "sac", "a2c", "dqn"):
            algo_prefix = f"{algo}_"
            if not rest.startswith(algo_prefix):
                continue
            mode_and_model = rest[len(algo_prefix) :]
            for mode in (
                "language_to_cost",
                "route_option_rank",
                "preference_scorer",
                "weighted_scorer",
                "graph_shortest",
                "no_llm",
            ):
                mode_prefix = f"{mode}_"
                if mode_and_model.startswith(mode_prefix):
                    model = mode_and_model[len(mode_prefix) :] or row.get("model", "") or "unknown"
                    scenario = row.get("scenario", "") or "semantic_constraint"
                    return scenario, algo, mode, _restore_model_tag(model)

    for scenario in ("semantic_constraint", "long_horizon"):
        marker = f"{scenario}_"
        start = label.find(marker)
        if start < 0:
            continue
        rest = label[start + len(marker) :]
        for algo in ("ppo", "sac", "a2c", "dqn"):
            algo_prefix = f"{algo}_"
            if not rest.startswith(algo_prefix):
                continue
            mode_and_model = rest[len(algo_prefix) :]
            for mode in (
                "llm_step_retry",
                "llm_step_order_ensemble",
                "llm_step",
                "llm_raw",
                "llm_retry",
                "graph_shortest",
                "greedy_progress",
                "greedy_hop",
                "greedy_risk",
                "random_legal",
                "no_llm",
                "llm",
            ):
                mode_prefix = f"{mode}_"
                if mode_and_model.startswith(mode_prefix):
                    model = mode_and_model[len(mode_prefix) :] or row.get("model", "") or "unknown"
                    if "_seed" in model:
                        model = model.split("_seed", 1)[0]
                    return scenario, algo, mode, model
    return (
        row.get("scenario", ""),
        row.get("algo", ""),
        row.get("planner_mode", ""),
        row.get("model", "") or "unknown",
    )


def _model_from_row(row: dict[str, str]) -> str:
    label = row.get("run_label", "")
    mode = row.get("planner_mode", "")
    if label:
        _scenario, _algo, _mode, model = _parse_run_label(label, row)
        if model and model != "unknown":
            return model
    if mode == "no_llm":
        return "no_llm"
    return row.get("model", "") or "unknown"


def _group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    scenario, algo, mode, model = _parse_run_label(row.get("run_label", ""), row)
    return (scenario, algo, mode, model)


def _normalize_run_summary(row: dict[str, str]) -> dict[str, Any]:
    scenario, algo, mode, model = _parse_run_label(row.get("run_label", ""), row)
    n = int(_float(row, "episodes"))
    success_rate = _float(row, "success_rate")
    strict_rate = _float(row, "strict_llm_plan_valid_rate")
    successes = round(success_rate * n)
    strict = round(strict_rate * n)
    success_lo, success_hi = _wilson(successes, n)
    strict_lo, strict_hi = _wilson(strict, n)
    out = {
        "scenario": scenario,
        "algo": algo,
        "planner_mode": mode,
        "model": model,
        "episodes": n,
        "success_rate": success_rate,
        "success_ci95_low": success_lo,
        "success_ci95_high": success_hi,
        "strict_valid_rate": strict_rate,
        "strict_ci95_low": strict_lo,
        "strict_ci95_high": strict_hi,
        "parse_ok_rate": _float(row, "parse_ok_rate"),
        "repair_rate": _float(row, "repaired_route_rate"),
        "collision_rate": _float(row, "collision_rate"),
        "timeout_rate": _float(row, "timeout_rate"),
        "invalid_route_rate": _float(row, "invalid_route_rate"),
        "mean_steps": _float(row, "mean_steps_all"),
        "mean_route_distance": _float(row, "mean_route_distance"),
        "mean_route_turns": _float(row, "mean_route_turns"),
        "mean_semantic_cost": _float(row, "mean_trajectory_semantic_cost"),
        "mean_route_score": _float(row, "mean_route_execution_score"),
        "mean_node_retention": _float(row, "mean_node_retention"),
        "mean_edit_distance": _float(row, "mean_edit_distance"),
        "mean_step_parse_ok": 0.0,
        "mean_step_count": 0.0,
        "raw_execution_rate": _float(row, "raw_llm_execution_rate"),
        "missing_edge_rate": _float(row, "failure_missing_edge_rate"),
        "missing_goal_rate": _float(row, "failure_missing_goal_rate"),
        "semantic_violation_rate": _float(row, "failure_semantic_violation_rate"),
        "llm_error_rate": _float(row, "failure_llm_error_rate"),
    }
    return out


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for (scenario, algo, mode, model), group in sorted(grouped.items()):
        n = len(group)
        successes = sum(int(_float(row, "success")) for row in group)
        strict = sum(int(_float(row, "plan_valid")) for row in group)
        repaired = sum(int(_float(row, "repaired")) for row in group)
        parsed = sum(int(_float(row, "parse_ok")) for row in group)
        collisions = sum(int(_float(row, "collision")) for row in group)
        timeouts = sum(int(_float(row, "timeout")) for row in group)
        success_lo, success_hi = _wilson(successes, n)
        strict_lo, strict_hi = _wilson(strict, n)
        row = {
            "scenario": scenario,
            "algo": algo,
            "planner_mode": mode,
            "model": model,
            "episodes": n,
            "success_rate": successes / n if n else 0.0,
            "success_ci95_low": success_lo,
            "success_ci95_high": success_hi,
            "strict_valid_rate": strict / n if n else 0.0,
            "strict_ci95_low": strict_lo,
            "strict_ci95_high": strict_hi,
            "parse_ok_rate": parsed / n if n else 0.0,
            "repair_rate": repaired / n if n else 0.0,
            "collision_rate": collisions / n if n else 0.0,
            "timeout_rate": timeouts / n if n else 0.0,
            "mean_steps": _mean([_float(r, "steps") for r in group]),
            "mean_route_distance": _mean([_float(r, "route_distance") for r in group if r.get("route_distance", "") != ""]),
            "mean_route_turns": _mean([_float(r, "route_turns") for r in group if r.get("route_turns", "") != ""]),
            "mean_semantic_cost": _mean([_float(r, "semantic_cost") for r in group if r.get("semantic_cost", "") != ""]),
            "mean_route_score": _mean([_float(r, "route_score") for r in group if r.get("route_score", "") != ""]),
            "mean_node_retention": _mean([_float(r, "node_retention") for r in group]),
            "mean_edit_distance": _mean([_float(r, "edit_distance") for r in group]),
            "mean_step_parse_ok": _mean([_float(r, "step_parse_ok_count") for r in group if r.get("step_parse_ok_count", "") != ""]),
            "mean_step_count": _mean([_float(r, "step_count") for r in group if r.get("step_count", "") != ""]),
            "raw_execution_rate": _mean([_float(r, "raw_execution") for r in group if r.get("raw_execution", "") != ""]),
            "missing_edge_rate": sum(1 for r in group if r.get("failure_category") == "missing_edge") / n if n else 0.0,
            "missing_goal_rate": sum(1 for r in group if r.get("failure_category") == "missing_goal") / n if n else 0.0,
            "semantic_violation_rate": sum(1 for r in group if r.get("failure_category") == "semantic_violation") / n if n else 0.0,
            "llm_error_rate": sum(1 for r in group if r.get("failure_category") == "llm_error") / n if n else 0.0,
        }
        out.append(row)
    return out


def merge_with_run_summaries(episode_summary: list[dict[str, Any]], run_summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (str(row["scenario"]), str(row["algo"]), str(row["planner_mode"]), str(row["model"])): row
        for row in episode_summary
    }
    for raw in run_summary_rows:
        if not raw.get("run_label"):
            continue
        row = _normalize_run_summary(raw)
        key = (str(row["scenario"]), str(row["algo"]), str(row["planner_mode"]), str(row["model"]))
        merged[key] = row
    return [merged[key] for key in sorted(merged)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv-glob", required=True)
    parser.add_argument("--run-summary-csv-glob", default="")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.episode_csv_glob)
    summary = summarize(rows)
    if args.run_summary_csv_glob:
        run_summary_rows = _read_rows(args.run_summary_csv_glob)
        summary = merge_with_run_summaries(summary, run_summary_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not summary:
        raise SystemExit(f"No rows matched {args.episode_csv_glob}")
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(
            f"{row['scenario']} {row['algo']} {row['planner_mode']} {row['model']}: "
            f"success={row['success_rate']:.3f} "
            f"ci95=[{row['success_ci95_low']:.3f},{row['success_ci95_high']:.3f}] "
            f"strict={row['strict_valid_rate']:.3f} repair={row['repair_rate']:.3f} "
            f"retention={row['mean_node_retention']:.3f}"
        )


if __name__ == "__main__":
    main()
