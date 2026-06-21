#!/usr/bin/env python3
"""Generate Paper A failure-taxonomy tables, report, and stacked-bar figure.

The goal is not to add another leaderboard. This script turns episode-level
CSV logs into an auditable failure taxonomy: high-level language/graph failures
versus downstream controller collisions/timeouts, plus order-gate fallback
usage when available.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TAXONOMY_ORDER = [
    "success",
    "parse_or_selection_failure",
    "missing_goal",
    "missing_edge",
    "semantic_plan_violation",
    "empty_route",
    "llm_runtime_error",
    "collision",
    "timeout",
    "other_invalid",
    "other_failure",
]

TAXONOMY_LABELS = {
    "success": "Success",
    "parse_or_selection_failure": "Parse/selection",
    "missing_goal": "Missing goal",
    "missing_edge": "Missing edge",
    "semantic_plan_violation": "Semantic plan",
    "empty_route": "Empty route",
    "llm_runtime_error": "LLM/runtime",
    "collision": "Collision",
    "timeout": "Timeout",
    "other_invalid": "Other invalid",
    "other_failure": "Other failure",
}

DETERMINISTIC_MODES = {
    "no_llm",
    "graph_shortest",
    "first_candidate",
    "weighted_scorer",
    "preference_scorer",
    "greedy_progress",
    "greedy_hop",
    "greedy_risk",
    "random_legal",
}

COLORS = {
    "success": "#009E73",
    "parse_or_selection_failure": "#D55E00",
    "missing_goal": "#E69F00",
    "missing_edge": "#CC79A7",
    "semantic_plan_violation": "#0072B2",
    "empty_route": "#999999",
    "llm_runtime_error": "#000000",
    "collision": "#56B4E9",
    "timeout": "#F0E442",
    "other_invalid": "#7A7A7A",
    "other_failure": "#B0B0B0",
}


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def read_rows(patterns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with open(path, newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        row["_source_csv"] = path
                        rows.append(row)
            except UnicodeDecodeError:
                with open(path, newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        row["_source_csv"] = path
                        rows.append(row)
    return rows


def parse_run_label(label: str, row: dict[str, str]) -> tuple[str, str, str, str]:
    scenario = row.get("scenario", "") or "unknown"
    algo = row.get("algo", "") or "unknown"
    mode = row.get("planner_mode", "") or "unknown"
    model = row.get("model", "") or "unknown"
    if not label:
        if mode in DETERMINISTIC_MODES:
            model = "no_llm"
        return scenario, algo, mode, model

    for candidate_scenario in ("semantic_constraint", "long_horizon"):
        marker = f"{candidate_scenario}_"
        idx = label.find(marker)
        if idx < 0:
            continue
        scenario = candidate_scenario
        rest = label[idx + len(marker) :]
        for candidate_algo in ("ppo", "sac", "a2c", "dqn"):
            prefix = f"{candidate_algo}_"
            if not rest.startswith(prefix):
                continue
            algo = candidate_algo
            tail = rest[len(prefix) :]
            modes = [
                "llm_step_consistency_gate",
                "llm_step_order_ensemble",
                "llm_step_retry",
                "llm_step",
                "llm_raw",
                "llm_retry",
                "preference_scorer",
                "weighted_scorer",
                "first_candidate",
                "graph_shortest",
                "greedy_progress",
                "greedy_hop",
                "greedy_risk",
                "random_legal",
                "no_llm",
                "llm",
            ]
            for candidate_mode in modes:
                mode_prefix = f"{candidate_mode}_"
                if tail.startswith(mode_prefix):
                    mode = candidate_mode
                    model = tail[len(mode_prefix) :] or model
                    model = model.split("_seed", 1)[0]
                    model = model.rsplit("_no_hop", 1)[0]
                    model = model.rsplit("_no_risk", 1)[0]
                    model = model.rsplit("_shuffle_order", 1)[0]
                    if mode in DETERMINISTIC_MODES:
                        model = "no_llm"
                    return scenario, algo, mode, model or "unknown"
    if mode in DETERMINISTIC_MODES:
        model = "no_llm"
    return scenario, algo, mode, model or "unknown"


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    scenario, algo, mode, model = parse_run_label(row.get("run_label", ""), row)
    perturb = "clean"
    if row.get("graph_edge_drop_rate") not in ("", None) or row.get("risk_center_noise") not in ("", None):
        edge = as_float(row, "graph_edge_drop_rate")
        noise = as_float(row, "risk_center_noise")
        radius = as_float(row, "risk_radius_scale", 1.0)
        if edge > 0 and (noise > 0 or abs(radius - 1.0) > 1e-9):
            perturb = "combined"
        elif edge > 0:
            perturb = "edge_drop"
        elif noise > 0 or abs(radius - 1.0) > 1e-9:
            perturb = "risk_noise"
    return scenario, algo, mode, model, perturb


def taxonomy(row: dict[str, str]) -> str:
    if as_int(row, "success"):
        return "success"

    mode = row.get("planner_mode", "")
    failure_category = (row.get("failure_category") or "").lower()
    failure_reason = (row.get("failure_reason") or "").lower()
    plan_valid = as_int(row, "plan_valid")
    parse_ok = as_int(row, "parse_ok")
    outcome = (row.get("outcome") or "").lower()
    is_llm = mode.startswith("llm")

    if plan_valid == 0 or outcome == "invalid_route" or failure_category not in ("", "ok", "no_llm_graph_search"):
        if failure_category == "llm_error" or failure_reason.startswith("llm_error"):
            return "llm_runtime_error"
        if failure_category == "missing_goal" or "missing_goal" in failure_reason:
            return "missing_goal"
        if failure_category == "missing_edge" or "missing_edge" in failure_reason:
            return "missing_edge"
        if failure_category == "semantic_violation" or "semantic_cost" in failure_reason:
            return "semantic_plan_violation"
        if failure_category == "empty" or failure_reason == "empty":
            return "empty_route"
        if is_llm and parse_ok == 0:
            return "parse_or_selection_failure"
        if outcome == "invalid_route":
            return "other_invalid"

    if as_int(row, "collision") or outcome == "collision":
        return "collision"
    if as_int(row, "timeout") or outcome == "timeout":
        return "timeout"
    return "other_failure"


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for (scenario, algo, mode, model, perturb), group in sorted(grouped.items()):
        n = len(group)
        counts = Counter(taxonomy(row) for row in group)
        gate_rows = [row for row in group if as_float(row, "order_gate_steps") > 0]
        gate_fallback_episodes = sum(1 for row in gate_rows if as_float(row, "order_gate_fallbacks") > 0)
        gate_accept_episodes = sum(1 for row in gate_rows if as_float(row, "order_gate_accepts") > 0)
        row_out: dict[str, Any] = {
            "scenario": scenario,
            "algo": algo,
            "planner_mode": mode,
            "model": model,
            "perturbation": perturb,
            "episodes": n,
            "success_rate": counts["success"] / n if n else 0.0,
            "gate_episode_count": len(gate_rows),
            "gate_fallback_episode_rate": gate_fallback_episodes / max(len(gate_rows), 1),
            "gate_accept_episode_rate": gate_accept_episodes / max(len(gate_rows), 1),
            "mean_order_gate_consistency": sum(as_float(row, "order_gate_mean_consistency") for row in gate_rows) / max(len(gate_rows), 1),
            "mean_semantic_cost": mean_value(group, "semantic_cost"),
            "mean_route_distance": mean_value(group, "route_distance"),
            "source_csv_count": len({row.get("_source_csv", "") for row in group}),
        }
        for cat in TAXONOMY_ORDER:
            row_out[f"{cat}_count"] = counts[cat]
            row_out[f"{cat}_rate"] = counts[cat] / n if n else 0.0
        out.append(row_out)
    return out


def mean_value(rows: list[dict[str, str]], key: str) -> float:
    vals = [as_float(row, key) for row in rows if row.get(key, "") not in ("", None)]
    return sum(vals) / len(vals) if vals else 0.0


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No summary rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def method_label(row: dict[str, Any]) -> str:
    model = pretty_model(str(row["model"]))
    mode = pretty_mode(str(row["planner_mode"]))
    scenario = "SC" if row["scenario"] == "semantic_constraint" else "LH"
    perturb = "" if row.get("perturbation") == "clean" else f"\n{row.get('perturbation')}"
    if model == "no_llm":
        return f"{scenario} {mode}{perturb}"
    return f"{scenario} {mode}\n{model}{perturb}"


def pretty_model(model: str) -> str:
    replacements = {
        "liquid_lfm2.5-1.2b": "LFM2.5-1.2B",
        "liquid_lfm2.5-1.2b": "LFM2.5-1.2B",
        "nvidia_nemotron-3-nano-4b": "Nemotron",
        "google_gemma-4-e4b": "Gemma-4-E4B",
        "google_gemma-3-1b": "Gemma-3-1B",
        "qwen_qwen3-1.7b": "Qwen3-1.7B",
        "lfm2.5-8b-a1b-mlx": "LFM2.5-8B-A1B",
    }
    return replacements.get(model, model)


def pretty_mode(mode: str) -> str:
    replacements = {
        "llm_step_consistency_gate": "gate",
        "llm_step_order_ensemble": "ensemble",
        "llm_step_retry": "retry",
        "llm_step": "step",
        "llm_raw": "raw",
        "no_llm": "graph",
        "graph_shortest": "shortest",
        "first_candidate": "first",
        "weighted_scorer": "weighted",
        "preference_scorer": "pref-score",
    }
    return replacements.get(mode, mode)


def selected_for_plot(rows: list[dict[str, Any]], max_bars: int) -> list[dict[str, Any]]:
    priority_modes = [
        "no_llm",
        "llm_raw",
        "llm_step",
        "llm_step_order_ensemble",
        "llm_step_consistency_gate",
        "first_candidate",
        "preference_scorer",
    ]
    priority_models = [
        "no_llm",
        "liquid_lfm2.5-1.2b",
        "nvidia_nemotron-3-nano-4b",
        "google_gemma-4-e4b",
        "lfm2.5-8b-a1b-mlx",
        "qwen_qwen3-1.7b",
        "google_gemma-3-1b",
    ]
    candidates = [row for row in rows if row["episodes"] >= 20 and row["planner_mode"] in set(priority_modes)]
    by_key = {
        (str(row["scenario"]), str(row["planner_mode"]), str(row["model"]), str(row.get("perturbation", "clean"))): row
        for row in candidates
    }
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(row: dict[str, Any] | None) -> None:
        if not row:
            return
        key = (str(row["scenario"]), str(row["planner_mode"]), str(row["model"]), str(row.get("perturbation", "clean")))
        if key not in seen:
            selected.append(row)
            seen.add(key)

    half = max(1, max_bars // 2)
    for scenario in ("long_horizon", "semantic_constraint"):
        before = len(selected)
        for mode in priority_modes:
            if len(selected) - before >= half:
                break
            if mode in DETERMINISTIC_MODES:
                add(by_key.get((scenario, mode, "no_llm", "clean")))
                continue
            for model in priority_models:
                if len(selected) - before >= half:
                    break
                add(by_key.get((scenario, mode, model, "clean")))

    if len(selected) < max_bars:
        candidates.sort(
            key=lambda row: (
                str(row.get("perturbation", "clean")) != "clean",
                str(row["scenario"]),
                priority_modes.index(str(row["planner_mode"])) if row["planner_mode"] in priority_modes else 99,
                priority_models.index(str(row["model"])) if row["model"] in priority_models else 99,
                -int(row["episodes"]),
            )
        )
        for row in candidates:
            if len(selected) >= max_bars:
                break
            add(row)
    return selected[:max_bars]


def write_figure(rows: list[dict[str, Any]], out_base: Path, max_bars: int) -> None:
    plot_rows = selected_for_plot(rows, max_bars)
    if not plot_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = [method_label(row) for row in plot_rows]
    fig_w = max(7.16, 0.46 * len(plot_rows))
    fig, ax = plt.subplots(figsize=(fig_w, 3.1), constrained_layout=True)
    bottoms = [0.0 for _ in plot_rows]
    x = list(range(len(plot_rows)))
    for cat in TAXONOMY_ORDER:
        vals = [100.0 * float(row.get(f"{cat}_rate", 0.0)) for row in plot_rows]
        if max(vals) <= 0:
            continue
        ax.bar(x, vals, bottom=bottoms, label=TAXONOMY_LABELS[cat], color=COLORS[cat], edgecolor="white", linewidth=0.25)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    ax.set_ylim(0, 100)
    ax.set_ylabel("Episode share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.38))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], path: Path, input_patterns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(int(row["episodes"]) for row in rows)
    overall_counts = Counter()
    for row in rows:
        for cat in TAXONOMY_ORDER:
            overall_counts[cat] += int(row[f"{cat}_count"])

    lines = [
        "# Paper A Failure Taxonomy Report",
        "",
        "This report is generated from episode-level CSV logs. It separates high-level route-interface failures from downstream controller outcomes and records order-gate fallback usage when available.",
        "",
        "## Inputs",
        "",
    ]
    for pattern in input_patterns:
        lines.append(f"- `{pattern}`")
    lines.extend(["", f"Total grouped rows: {len(rows)}", f"Total episodes: {total}", "", "## Overall Episode Taxonomy", ""])
    lines.append("| Category | Episodes | Rate |")
    lines.append("|---|---:|---:|")
    for cat in TAXONOMY_ORDER:
        count = overall_counts[cat]
        if count == 0:
            continue
        lines.append(f"| {TAXONOMY_LABELS[cat]} | {count} | {count / max(total, 1):.1%} |")

    lines.extend(["", "## Highest Non-Success Failure Rates", ""])
    lines.append("| Scenario | Planner | Model | Perturb. | Episodes | Dominant non-success | Rate | Success |")
    lines.append("|---|---|---|---|---:|---|---:|---:|")
    ranked = []
    for row in rows:
        non_success = [(cat, float(row[f"{cat}_rate"])) for cat in TAXONOMY_ORDER if cat != "success"]
        cat, rate = max(non_success, key=lambda item: item[1])
        ranked.append((rate, cat, row))
    ranked.sort(key=lambda item: (item[0], item[1], str(item[2].get("planner_mode", "")), str(item[2].get("model", ""))), reverse=True)
    for rate, cat, row in ranked[:20]:
        if rate <= 0:
            continue
        lines.append(
            f"| {row['scenario']} | `{row['planner_mode']}` | {row['model']} | {row['perturbation']} | "
            f"{row['episodes']} | {TAXONOMY_LABELS[cat]} | {rate:.1%} | {float(row['success_rate']):.1%} |"
        )

    gate_rows = [row for row in rows if int(row.get("gate_episode_count", 0)) > 0]
    if gate_rows:
        lines.extend(["", "## Order-Gate Diagnostics", ""])
        lines.append("| Scenario | Planner | Model | Perturb. | Gate episodes | Fallback episode rate | Accept episode rate | Mean consistency | Success |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        gate_rows.sort(key=lambda row: (-float(row["gate_fallback_episode_rate"]), row["scenario"], row["model"]))
        for row in gate_rows[:20]:
            lines.append(
                f"| {row['scenario']} | `{row['planner_mode']}` | {row['model']} | {row['perturbation']} | "
                f"{row['gate_episode_count']} | {float(row['gate_fallback_episode_rate']):.1%} | "
                f"{float(row['gate_accept_episode_rate']):.1%} | {float(row['mean_order_gate_consistency']):.2f} | "
                f"{float(row['success_rate']):.1%} |"
            )

    lines.extend(
        [
            "",
            "## Writing Guidance",
            "",
            "- Use high-level categories such as parse/selection failure, missing goal, missing edge, and semantic plan violation to discuss route-interface limits.",
            "- Use collision and timeout to discuss downstream frozen-controller limits; do not attribute them only to the LLM.",
            "- If gate fallback rates are high, frame the gate as an uncertainty detector and safety fallback rather than as evidence that the LLM itself controlled most successful routes.",
            "- Re-run this script after each new experiment batch so the figure and report stay synchronized with the logs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure taxonomy from Paper A episode CSV logs.")
    parser.add_argument(
        "--episode-csv-glob",
        action="append",
        default=[],
        help="Glob for episode CSVs. Can be supplied multiple times.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("paper_assets/paper_a/failure_taxonomy"))
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--figure-base", type=Path)
    parser.add_argument("--max-bars", type=int, default=18)
    parser.add_argument("--run-label-regex", default="")
    args = parser.parse_args()

    patterns = args.episode_csv_glob or ["paper_assets/paper_a/rerun_logs/**/episodes/*.csv"]
    rows = read_rows(patterns)
    if args.run_label_regex:
        rx = re.compile(args.run_label_regex)
        rows = [row for row in rows if rx.search(row.get("run_label", "")) or rx.search(row.get("_source_csv", ""))]
    if not rows:
        raise SystemExit("No episode rows matched the requested glob/regex")

    summary = summarize(rows)
    summary_csv = args.summary_csv or args.out_dir / "failure_taxonomy_summary.csv"
    report_md = args.report_md or args.out_dir / "failure_taxonomy_report.md"
    figure_base = args.figure_base or args.out_dir / "failure_taxonomy_stacked"

    write_csv(summary, summary_csv)
    write_report(summary, report_md, patterns)
    write_figure(summary, figure_base, args.max_bars)

    print(f"rows={len(rows)} groups={len(summary)}")
    print(f"summary_csv={summary_csv}")
    print(f"report_md={report_md}")
    if figure_base.with_suffix(".pdf").exists():
        print(f"figure_pdf={figure_base.with_suffix('.pdf')}")
        print(f"figure_png={figure_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
