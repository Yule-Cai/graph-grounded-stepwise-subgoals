#!/usr/bin/env python3
"""Generate the supplement figure for the 250-episode preference extension."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY = Path(
    "paper_assets/paper_a/rerun_logs/preference_250_extended_models_20260620/"
    "language_cost_option_summary_with_ci.csv"
)
FIG_DIR = Path("artifacts/generated_figures")
OUT_STEM = "fig_supp_preference_250_extension"

PALETTE = {
    "language_to_cost": "#0072B2",
    "route_option_rank": "#D55E00",
    "grid": "#D8DEE9",
    "text": "#1F2933",
}
MARKERS = {
    "language_to_cost": "o",
    "route_option_rank": "s",
}


def short_model(name: str) -> str:
    mapping = {
        "google/gemma-3-1b": "Gemma-3\n1B",
        "google/gemma-4-12b": "Gemma-4\n12B",
        "google/gemma-4-e4b": "Gemma-4\nE4B",
        "qwen/qwen3-1.7b": "Qwen3\n1.7B",
        "qwenpaw-flash-9b": "Qwenpaw\n9B",
    }
    return mapping.get(name, name.replace("/", "\n"))


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.95},
    )


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)


def should_annotate(row: pd.Series, panel: str) -> bool:
    model = str(row["model"])
    mode = str(row["planner_mode"])
    if panel == "A":
        return (
            model in {"google/gemma-3-1b", "qwenpaw-flash-9b"}
            or (mode == "language_to_cost" and model == "google/gemma-4-12b")
            or (mode == "route_option_rank" and model == "google/gemma-4-12b")
        )
    return (
        model in {"google/gemma-3-1b", "qwenpaw-flash-9b"}
        or (mode == "route_option_rank" and model == "google/gemma-4-12b")
    )


def label_offset(row: pd.Series, panel: str) -> tuple[int, int]:
    model = str(row["model"])
    mode = str(row["planner_mode"])
    if panel == "A" and model == "google/gemma-4-12b" and mode == "language_to_cost":
        return (-34, -20)
    if panel == "A" and model == "google/gemma-4-12b":
        return (5, -18)
    if panel == "A" and model == "qwenpaw-flash-9b":
        return (5, 4)
    if panel == "B" and model == "google/gemma-4-12b":
        return (5, -2)
    if panel == "B" and model == "google/gemma-3-1b":
        return (6, -8)
    return (5, 4)


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    panel: str,
) -> None:
    for mode, sub in df.groupby("planner_mode"):
        ax.scatter(
            100.0 * sub[xcol],
            100.0 * sub[ycol] if ycol.endswith("_rate") else sub[ycol],
            s=72,
            marker=MARKERS[mode],
            color=PALETTE[mode],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.94,
            label=mode.replace("_", " "),
            zorder=3,
        )
    for _, row in df.iterrows():
        if should_annotate(row, panel):
            x = 100.0 * row[xcol]
            y = 100.0 * row[ycol] if ycol.endswith("_rate") else row[ycol]
            dx, dy = label_offset(row, panel)
            ax.annotate(
                short_model(str(row["model"])),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.8,
                color=PALETTE["text"],
                alpha=0.86,
            )
    style_axes(ax, xlabel, ylabel)


def main() -> None:
    setup()
    df = pd.read_csv(SUMMARY)
    expected = {"success_rate", "semantic_violation_rate", "strict_valid_rate", "mean_semantic_cost"}
    missing = expected.difference(df.columns)
    if missing:
        raise SystemExit(f"missing required columns in {SUMMARY}: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(7.18, 2.72), constrained_layout=True)

    plot_panel(
        axes[0],
        df,
        "success_rate",
        "semantic_violation_rate",
        "Execution success (%)",
        "Semantic violation (%)",
        panel="A",
    )
    axes[0].set_xlim(70, 90)
    axes[0].set_ylim(40, 102)
    axes[0].axhspan(80, 102, color="#FDE0DD", alpha=0.38, zorder=0)
    axes[0].text(70.5, 99.0, "high violation", fontsize=6.8, color="#8B1A1A", va="top")
    add_panel_label(axes[0], "A")

    plot_panel(
        axes[1],
        df,
        "strict_valid_rate",
        "mean_semantic_cost",
        "Strict validity (%)",
        "Semantic cost",
        panel="B",
    )
    axes[1].set_xlim(-3, 56)
    axes[1].set_ylim(0.09, 0.178)
    axes[1].axvspan(-3, 20, color="#FDE0DD", alpha=0.34, zorder=0)
    axes[1].text(0, 0.174, "low strictness", fontsize=6.8, color="#8B1A1A", va="top")
    add_panel_label(axes[1], "B")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.06),
        handletextpad=0.5,
        columnspacing=1.8,
    )

    pdf = FIG_DIR / f"{OUT_STEM}.pdf"
    png = FIG_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
