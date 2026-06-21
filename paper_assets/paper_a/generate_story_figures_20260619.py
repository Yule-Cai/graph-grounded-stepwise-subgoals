#!/usr/bin/env python3
"""Generate story-driven Paper A figures from the latest completed logs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("paper_assets/paper_a")
AAAI_FIG_DIR = Path("artifacts/generated_figures")
ORDER = ROOT / "rerun_logs/order_gate_threshold_sweep_20260618/order_gate_threshold_sweep_summary_with_ci.csv"
MAIN = ROOT / "rerun_logs/final_aaai_stepwise_20260602_050932/final_experiment_summary_lmstudio_only.csv"
DETERMINISTIC = ROOT / "rerun_logs/smoke_deterministic_scorers_20260612/deterministic_scorer_summary_with_ci.csv"
GATE_DIR = ROOT / "rerun_logs/graph_perturbation_llm_gate_20260618/summaries"
BASE_DIR = ROOT / "rerun_logs/graph_perturbation_baselines_20260617/summaries"


PALETTE = {
    "graph": "#4D4D4D",
    "raw": "#D55E00",
    "step": "#009E73",
    "strict": "#0072B2",
    "shuffle": "#CC79A7",
    "ensemble": "#56B4E9",
    "gate": "#E69F00",
    "pref": "#6A3D9A",
    "grid": "#D8DEE9",
    "text": "#1F2933",
}


def pct(x):
    return 100.0 * x


def setup() -> None:
    AAAI_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(AAAI_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(AAAI_FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.95},
    )


def style_rate_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)


def order_label(model: str, planner_mode: str) -> str:
    if planner_mode == "no_llm":
        return "Graph route"
    if "shuffle_order" in model:
        return "Shuffled order"
    if "order_ensemble" in model or planner_mode == "llm_step_order_ensemble":
        return "Order ensemble"
    if "gate_lenient" in model:
        return "Gate, lenient"
    if "gate_balanced" in model:
        return "Gate, balanced"
    if "gate_strict" in model:
        return "Gate, strict"
    if "canonical" in model:
        return "Stepwise"
    return model


def get_order_row(df: pd.DataFrame, scenario: str, label: str) -> pd.Series:
    tmp = df[df["scenario"].eq(scenario)].copy()
    tmp["label"] = [order_label(m, p) for m, p in zip(tmp["model"], tmp["planner_mode"], strict=False)]
    match = tmp[tmp["label"].eq(label)]
    if match.empty:
        raise KeyError((scenario, label))
    return match.iloc[0]


def main_story_figure() -> None:
    main = pd.read_csv(MAIN)
    order = pd.read_csv(ORDER)
    det = pd.read_csv(DETERMINISTIC) if DETERMINISTIC.exists() else pd.DataFrame()

    fig = plt.figure(figsize=(7.25, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.04], wspace=0.46, hspace=0.54)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A: Planning granularity.
    models = [
        ("liquid_lfm2.5-1.2b", "LFM2.5\n1.2B"),
        ("nvidia_nemotron-3-nano-4b", "Nemotron\n4B"),
    ]
    x = np.arange(len(models))
    width = 0.24
    raw, step, strict = [], [], []
    for model, _ in models:
        raw_row = main[
            main["scenario"].eq("long_horizon")
            & main["planner_mode"].eq("llm_raw")
            & main["model"].eq(model)
        ].iloc[0]
        step_row = main[
            main["scenario"].eq("long_horizon")
            & main["planner_mode"].eq("llm_step")
            & main["model"].eq(model)
        ].iloc[0]
        raw.append(pct(raw_row["success_rate"]))
        step.append(pct(step_row["success_rate"]))
        strict.append(pct(step_row["strict_valid_rate"]))
    ax_a.bar(x - width, raw, width, color=PALETTE["raw"], label="raw")
    ax_a.bar(x, step, width, color=PALETTE["step"], label="step")
    ax_a.bar(x + width, strict, width, color=PALETTE["strict"], label="strict")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([m[1] for m in models])
    style_rate_axis(ax_a)
    ax_a.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.56, 1.045),
        ncol=3,
        handlelength=1.1,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    for group_x, vals in zip((x - width, x, x + width), (raw, step, strict), strict=False):
        for xx, val in zip(group_x, vals, strict=False):
            ax_a.text(xx, val + 2.4, f"{val:.0f}", ha="center", fontsize=7, zorder=5)
    add_panel_label(ax_a, "A")

    # B: Order sensitivity and gated fallback.
    labels = ["Graph route", "Stepwise", "Shuffled order", "Order ensemble", "Gate, strict"]
    display_labels = ["Graph route", "Stepwise", "Shuffled order", "Order ensemble", "Gate strict\n(fallback)"]
    colors = [PALETTE["graph"], PALETTE["step"], PALETTE["shuffle"], PALETTE["ensemble"], PALETTE["gate"]]
    y = np.arange(len(labels))
    offsets = {"long_horizon": 0.13, "semantic_constraint": -0.13}
    scen_names = {"long_horizon": "Long", "semantic_constraint": "Semantic"}
    markers = {"long_horizon": "o", "semantic_constraint": "s"}
    for scenario in ["long_horizon", "semantic_constraint"]:
        vals, xerr_low, xerr_high = [], [], []
        for label in labels:
            row = get_order_row(order, scenario, label)
            vals.append(pct(row["success_rate"]))
            xerr_low.append(pct(row["success_rate"] - row["success_ci95_low"]))
            xerr_high.append(pct(row["success_ci95_high"] - row["success_rate"]))
        ax_b.errorbar(
            vals,
            y + offsets[scenario],
            xerr=[xerr_low, xerr_high],
            fmt=markers[scenario],
            markersize=4.5,
            color="#1F2933",
            ecolor="#8B95A1",
            elinewidth=1.0,
            capsize=2.2,
            label=scen_names[scenario],
            zorder=3,
        )
    for yy, color in zip(y, colors, strict=False):
        ax_b.axhspan(yy - 0.42, yy + 0.42, color=color, alpha=0.07, zorder=0)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(display_labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 92)
    ax_b.set_xlabel("Success rate (%)")
    ax_b.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax_b.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        handlelength=1.2,
    )
    add_panel_label(ax_b, "B")

    # C: Semantic success-exposure diagnostic.
    methods = [
        ("Graph route", "semantic_constraint", "Graph", PALETTE["graph"], "o"),
        ("Stepwise", "semantic_constraint", "Stepwise", PALETTE["step"], "o"),
        ("Shuffled order", "semantic_constraint", "Shuffled", PALETTE["shuffle"], "X"),
        ("Order ensemble", "semantic_constraint", "Ensemble", PALETTE["ensemble"], "D"),
        ("Gate, strict", "semantic_constraint", "Gate", PALETTE["gate"], "s"),
    ]
    points = []
    for label, scenario, text, color, marker in methods:
        row = get_order_row(order, scenario, label)
        points.append((pct(row["success_rate"]), float(row["mean_semantic_cost"]), text, color, marker))
    if not det.empty:
        for planner, text, color in [
            ("first_candidate", "First cand.", "#7F7F7F"),
            ("preference_scorer", "Pref. scorer", PALETTE["pref"]),
        ]:
            match = det[
                det["scenario"].eq("semantic_constraint")
                & det["planner_mode"].eq(planner)
            ]
            if not match.empty:
                row = match.iloc[0]
                points.append((pct(row["success_rate"]), float(row["mean_semantic_cost"]), text, color, "^"))
    for sx, sy, text, color, marker in points:
        ax_c.scatter(
            sx,
            sy,
            s=38,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=text,
        )
    ax_c.set_xlim(0, 92)
    ax_c.set_ylim(0, 0.175)
    ax_c.set_xlabel("Semantic benchmark success (%)")
    ax_c.set_ylabel("Executed semantic cost")
    ax_c.grid(color=PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax_c.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="upper left",
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.25,
        borderpad=0.25,
    )
    add_panel_label(ax_c, "C")

    # D: Conservative gate threshold sweep.
    gate_labels = ["Stepwise", "Lenient", "Balanced", "Strict", "Graph route"]
    gate_map = {
        "Stepwise": "Stepwise",
        "Lenient": "Gate, lenient",
        "Balanced": "Gate, balanced",
        "Strict": "Gate, strict",
        "Graph route": "Graph route",
    }
    xx = np.arange(len(gate_labels))
    for scenario, color, marker in [
        ("long_horizon", PALETTE["strict"], "o"),
        ("semantic_constraint", PALETTE["raw"], "s"),
    ]:
        vals = [pct(get_order_row(order, scenario, gate_map[g])["success_rate"]) for g in gate_labels]
        ax_d.plot(xx, vals, marker=marker, color=color, linewidth=1.8, label=scen_names[scenario])
    ax_d.set_xticks(xx)
    ax_d.set_xticklabels(["step", "len.", "bal.", "strict", "graph"])
    style_rate_axis(ax_d)
    ax_d.legend(frameon=False, loc="lower right")
    ax_d.set_xlabel("Route-interface setting")
    add_panel_label(ax_d, "D")

    save(fig, "fig_main_story_results")


def parse_summary_name(path: Path) -> tuple[str, str, str] | None:
    pat = re.compile(
        r"^(?P<map>.+?)_(?P<scenario>long_horizon|semantic_constraint)_(?P<perturb>clean|edge_drop|risk_noise|combined)_ppo_"
    )
    m = pat.match(path.name)
    if not m:
        return None
    return m.group("map"), m.group("scenario"), m.group("perturb")


def read_summaries(directory: Path, mode_filter: str | None = None) -> pd.DataFrame:
    rows = []
    for path in directory.glob("*.csv"):
        parsed = parse_summary_name(path)
        if not parsed:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.rename(
            columns={
                "strict_llm_plan_valid_rate": "strict_valid_rate",
                "mean_trajectory_semantic_cost": "mean_semantic_cost",
            }
        )
        row = df.iloc[0].to_dict()
        map_name, scenario, perturb = parsed
        row["map"] = map_name
        row["scenario_from_file"] = scenario
        row["perturbation"] = perturb
        if mode_filter and row.get("planner_mode") != mode_filter:
            continue
        rows.append(row)
    return pd.DataFrame(rows)


def perturbation_heatmap() -> None:
    gate = read_summaries(GATE_DIR, "llm_step_consistency_gate")
    base = read_summaries(BASE_DIR, "no_llm")
    if gate.empty or base.empty:
        return
    key = ["map", "scenario_from_file", "perturbation"]
    merged = gate.merge(base, on=key, suffixes=("_gate", "_graph"))
    merged["delta_success"] = pct(merged["success_rate_gate"] - merged["success_rate_graph"])
    merged["delta_cost"] = merged["mean_semantic_cost_gate"] - merged["mean_semantic_cost_graph"]
    maps = ["reference_family_flat", "reference_villa_ground", "studio_apartment"]
    perts = ["clean", "edge_drop", "risk_noise", "combined"]
    rows = [(m, p) for m in maps for p in perts]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 4.7), gridspec_kw={"width_ratios": [1, 1, 1.08]})
    panels = [
        ("long_horizon", "delta_success", "Long success\nΔ gate - graph", -25, 25, "RdBu"),
        ("semantic_constraint", "delta_success", "Semantic success\nΔ gate - graph", -25, 25, "RdBu"),
        ("semantic_constraint", "delta_cost", "Semantic cost\nΔ gate - graph", -0.08, 0.08, "PuOr_r"),
    ]
    for ax, (scenario, metric, title, vmin, vmax, cmap) in zip(axes, panels, strict=False):
        mat = np.full((len(rows), 1), np.nan)
        text = []
        for i, (m, p) in enumerate(rows):
            match = merged[
                merged["map"].eq(m)
                & merged["scenario_from_file"].eq(scenario)
                & merged["perturbation"].eq(p)
            ]
            if not match.empty:
                mat[i, 0] = match.iloc[0][metric]
                text.append(mat[i, 0])
            else:
                text.append(np.nan)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([0])
        ax.set_xticklabels(["gate - graph"])
        ax.set_yticks(np.arange(len(rows)))
        if ax is axes[0]:
            ax.set_yticklabels([f"{m.replace('_', ' ')}\n{p.replace('_', ' ')}" for m, p in rows])
        else:
            ax.set_yticklabels([])
        for i, val in enumerate(text):
            if np.isfinite(val):
                label = f"{val:+.0f}" if metric == "delta_success" else f"{val:+.3f}"
                ax.text(0, i, label, ha="center", va="center", fontsize=6.7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.065, pad=0.04)
        cbar.ax.tick_params(labelsize=6.5)
    save(fig, "fig_supp_perturbation_gate_heatmap")


def gate_threshold_figure() -> None:
    df = pd.read_csv(ORDER)
    labels = ["Stepwise", "Lenient", "Balanced", "Strict", "Graph route"]
    label_map = {
        "Stepwise": "Stepwise",
        "Lenient": "Gate, lenient",
        "Balanced": "Gate, balanced",
        "Strict": "Gate, strict",
        "Graph route": "Graph route",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.65), sharey=True)
    for ax, scenario, title in zip(axes, ["long_horizon", "semantic_constraint"], ["Long horizon", "Semantic constraint"], strict=False):
        x = np.arange(len(labels))
        succ = [pct(get_order_row(df, scenario, label_map[l])["success_rate"]) for l in labels]
        strict = [pct(get_order_row(df, scenario, label_map[l])["strict_valid_rate"]) for l in labels]
        parse = [
            np.nan if l == "Graph route" else pct(get_order_row(df, scenario, label_map[l])["parse_ok_rate"])
            for l in labels
        ]
        ax.plot(x, succ, marker="o", color=PALETTE["step"], label="success", linewidth=1.8)
        ax.plot(x, strict, marker="s", color=PALETTE["strict"], label="strict valid", linewidth=1.5)
        ax.plot(x, parse, marker="^", color=PALETTE["gate"], label="LLM parse/accept", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["step", "len.", "bal.", "strict", "graph"])
        style_rate_axis(ax)
        ax.set_xlabel("Gate setting")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, "fig_supp_gate_threshold_sweep")


def main_story_figure() -> None:
    """Publication-facing Figure 2 with less text collision and stronger visual hierarchy."""
    main = pd.read_csv(MAIN)
    order = pd.read_csv(ORDER)

    fig = plt.figure(figsize=(7.25, 4.72))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 1.0],
        height_ratios=[1.0, 1.08],
        wspace=0.48,
        hspace=0.64,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A: planning granularity as a compact point-range plot instead of crowded bars.
    models = [
        ("liquid_lfm2.5-1.2b", "LFM2.5-1.2B"),
        ("nvidia_nemotron-3-nano-4b", "Nemotron-4B"),
    ]
    rows = []
    for model, label in models:
        raw_row = main[
            main["scenario"].eq("long_horizon")
            & main["planner_mode"].eq("llm_raw")
            & main["model"].eq(model)
        ].iloc[0]
        step_row = main[
            main["scenario"].eq("long_horizon")
            & main["planner_mode"].eq("llm_step")
            & main["model"].eq(model)
        ].iloc[0]
        rows.append(
            {
                "label": label,
                "raw": pct(raw_row["success_rate"]),
                "step": pct(step_row["success_rate"]),
                "strict": pct(step_row["strict_valid_rate"]),
            }
        )
    y = np.arange(len(rows))[::-1]
    for yy, row in zip(y, rows, strict=False):
        ax_a.plot(
            [row["raw"], row["step"], row["strict"]],
            [yy, yy, yy],
            color="#CBD5E1",
            linewidth=2.2,
            zorder=1,
        )
        ax_a.scatter(row["raw"], yy, s=42, color=PALETTE["raw"], marker="o", label="raw" if yy == y[0] else None, zorder=3)
        ax_a.scatter(row["step"], yy, s=52, color=PALETTE["step"], marker="s", label="step" if yy == y[0] else None, zorder=3)
        ax_a.scatter(row["strict"], yy, s=58, color=PALETTE["strict"], marker="D", label="strict" if yy == y[0] else None, zorder=3)
        val_dy = -0.20 if yy == y[0] else 0.18
        val_va = "top" if yy == y[0] else "bottom"
        ax_a.text(row["raw"] + 2.3, yy + val_dy, f"{row['raw']:.0f}", fontsize=6.8, ha="left", va=val_va)
        ax_a.text(row["step"] + 1.8, yy + val_dy, f"{row['step']:.0f}", fontsize=6.8, ha="left", va=val_va)
        ax_a.text(row["strict"] + 1.8, yy + val_dy, f"{row['strict']:.0f}", fontsize=6.8, ha="left", va=val_va)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([r["label"] for r in rows])
    ax_a.set_xlim(-2.0, 105)
    ax_a.set_xlabel("Long-horizon rate (%)")
    ax_a.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax_a.set_axisbelow(True)
    ax_a.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    add_panel_label(ax_a, "A")

    # B: order sensitivity as a forest plot with confidence intervals.
    labels = ["Graph route", "Stepwise", "Shuffled order", "Order ensemble", "Gate, strict"]
    display_labels = ["Graph route", "Stepwise", "Shuffled", "Ensemble", "Strict gate"]
    yb = np.arange(len(labels))
    offsets = {"long_horizon": -0.12, "semantic_constraint": 0.12}
    scen_names = {"long_horizon": "Long", "semantic_constraint": "Semantic"}
    markers = {"long_horizon": "o", "semantic_constraint": "s"}
    colors = {"long_horizon": PALETTE["strict"], "semantic_constraint": PALETTE["raw"]}
    for yy, color in zip(yb, [PALETTE["graph"], PALETTE["step"], PALETTE["shuffle"], PALETTE["ensemble"], PALETTE["gate"]], strict=False):
        ax_b.axhspan(yy - 0.40, yy + 0.40, color=color, alpha=0.055, zorder=0)
    for scenario in ["long_horizon", "semantic_constraint"]:
        vals, xerr_low, xerr_high = [], [], []
        for label in labels:
            row = get_order_row(order, scenario, label)
            vals.append(pct(row["success_rate"]))
            xerr_low.append(pct(row["success_rate"] - row["success_ci95_low"]))
            xerr_high.append(pct(row["success_ci95_high"] - row["success_rate"]))
        ax_b.errorbar(
            vals,
            yb + offsets[scenario],
            xerr=[xerr_low, xerr_high],
            fmt=markers[scenario],
            markersize=4.3,
            color=colors[scenario],
            ecolor="#94A3B8",
            elinewidth=0.95,
            capsize=2.0,
            label=scen_names[scenario],
            zorder=3,
        )
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(display_labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 92)
    ax_b.set_xlabel("Success rate (%)")
    ax_b.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax_b.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.99, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    add_panel_label(ax_b, "B")

    # C: semantic trade-off with direct labels and a clear preferred region.
    points = []
    for label, text, color, marker in [
        ("Graph route", "graph", PALETTE["graph"], "o"),
        ("Stepwise", "step", PALETTE["step"], "o"),
        ("Shuffled order", "shuffle", PALETTE["shuffle"], "X"),
        ("Order ensemble", "ensemble", PALETTE["ensemble"], "D"),
        ("Gate, strict", "gate", PALETTE["gate"], "s"),
    ]:
        row = get_order_row(order, "semantic_constraint", label)
        points.append((pct(row["success_rate"]), float(row["mean_semantic_cost"]), text, color, marker))
    ax_c.axvspan(60, 92, ymin=0.0, ymax=0.52, color="#ECFDF5", alpha=0.75, zorder=0)
    ax_c.axhspan(0.0, 0.09, xmin=0.63, xmax=1.0, color="#DCFCE7", alpha=0.55, zorder=0)
    for sx, sy, text, color, marker in points:
        ax_c.scatter(sx, sy, s=46, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
        label_pos = {
            "graph": (80.5, 0.091, "left"),
            "step": (70.0, 0.106, "left"),
            "shuffle": (15.0, 0.030, "left"),
            "ensemble": (42.5, 0.102, "left"),
            "gate": (78.2, 0.066, "left"),
        }
        tx, ty, ha = label_pos[text]
        ax_c.text(tx, ty, text, fontsize=6.8, color=PALETTE["text"], ha=ha, va="center")
    ax_c.text(69, 0.012, "preferred region", fontsize=6.8, color="#166534", ha="center", va="bottom")
    ax_c.set_xlim(0, 92)
    ax_c.set_ylim(0, 0.165)
    ax_c.set_xlabel("Semantic benchmark success (%)")
    ax_c.set_ylabel("Executed semantic cost")
    ax_c.grid(color=PALETTE["grid"], linewidth=0.6, alpha=0.80)
    add_panel_label(ax_c, "C")

    # D: same-axis success and strict validity under increasingly conservative gates.
    gate_labels = ["Stepwise", "Lenient", "Balanced", "Strict", "Graph route"]
    gate_map = {
        "Stepwise": "Stepwise",
        "Lenient": "Gate, lenient",
        "Balanced": "Gate, balanced",
        "Strict": "Gate, strict",
        "Graph route": "Graph route",
    }
    xd = np.arange(len(gate_labels))
    for scenario, color, marker in [
        ("long_horizon", PALETTE["strict"], "o"),
        ("semantic_constraint", PALETTE["raw"], "s"),
    ]:
        succ = [pct(get_order_row(order, scenario, gate_map[g])["success_rate"]) for g in gate_labels]
        strict = [pct(get_order_row(order, scenario, gate_map[g])["strict_valid_rate"]) for g in gate_labels]
        ax_d.plot(xd, succ, marker=marker, color=color, linewidth=1.8, label=f"{scen_names[scenario]} success")
        ax_d.plot(xd, strict, marker=marker, color=color, linewidth=1.2, linestyle="--", alpha=0.72, label=f"{scen_names[scenario]} strict")
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(["step", "len.", "bal.", "strict", "graph"])
    ax_d.set_ylim(0, 105)
    ax_d.set_ylabel("Rate (%)")
    ax_d.set_xlabel("Route-interface setting")
    ax_d.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax_d.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=2,
        fontsize=6.5,
        handlelength=1.7,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    add_panel_label(ax_d, "D")

    save(fig, "fig_main_story_results")


def main() -> None:
    setup()
    main_story_figure()
    perturbation_heatmap()
    gate_threshold_figure()


if __name__ == "__main__":
    main()
