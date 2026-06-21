#!/usr/bin/env python3
"""Generate Paper A figures for the current stepwise-subgoal draft."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path("paper_assets/paper_a")
SUMMARY = ROOT / "rerun_logs/final_aaai_stepwise_20260602_050932/final_experiment_summary_lmstudio_only.csv"
FIG_DIR = ROOT / "figures"


MODEL_LABELS = {
    "lfm2.5-8b-a1b-mlx": "LFM2.5-8B",
    "nvidia_nemotron-3-nano-4b": "Nemotron-4B",
    "qwen_qwen3-1.7b": "Qwen3-1.7B",
    "google_gemma-3-1b": "Gemma-3-1B",
    "google_gemma-4-e4b": "Gemma-4-E4B",
    "liquid_lfm2.5-1.2b": "LFM2.5-1.2B",
    "no_llm": "No LLM",
}

COLORS = {
    "blue": "#3368a8",
    "green": "#2f7d59",
    "orange": "#c96f2d",
    "red": "#b84a4a",
    "purple": "#7256a5",
    "gray": "#596579",
    "light": "#f6f8fb",
}


def load_rows() -> list[dict[str, str]]:
    with SUMMARY.open() as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def rows_for(rows: list[dict[str, str]], scenario: str, mode: str, algo: str = "ppo") -> list[dict[str, str]]:
    return [r for r in rows if r["scenario"] == scenario and r["planner_mode"] == mode and r["algo"] == algo]


def find_row(rows: list[dict[str, str]], scenario: str, mode: str, model: str, algo: str = "ppo") -> dict[str, str] | None:
    for row in rows:
        if row["scenario"] == scenario and row["planner_mode"] == mode and row["model"] == model and row["algo"] == algo:
            return row
    return None


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.7, alpha=0.9)


def add_box(ax, xy, wh, text, color, fontsize=9):
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.035,rounding_size=0.025",
        linewidth=1.4,
        facecolor=color,
        edgecolor="#2f3b4f",
    )
    ax.add_patch(box)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold")


def add_arrow(ax, start, end, color="#2f3b4f"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color))


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, (0.02, 0.30), (0.15, 0.42), "ROS2/Gazebo\nsemantic world", "#eaf2fb")
    add_box(ax, (0.23, 0.58), (0.18, 0.25), "Compact graph\nnodes + legal edges", "#edf7f0")
    add_box(ax, (0.23, 0.17), (0.18, 0.25), "Task goal +\nsemantic preference", "#fff7e8")
    add_box(ax, (0.48, 0.33), (0.20, 0.38), "Edge LLM\nnext-subgoal choice\n(llm_step)", "#edf7f0")
    add_box(ax, (0.75, 0.57), (0.20, 0.25), "Validated waypoint\nsequence", "#edf7f0")
    add_box(ax, (0.75, 0.18), (0.20, 0.25), "Frozen PPO/SAC\nlocal controller", "#efeafb")
    add_arrow(ax, (0.17, 0.52), (0.23, 0.70))
    add_arrow(ax, (0.17, 0.48), (0.23, 0.30))
    add_arrow(ax, (0.41, 0.70), (0.48, 0.55))
    add_arrow(ax, (0.41, 0.30), (0.48, 0.45))
    add_arrow(ax, (0.68, 0.53), (0.75, 0.70), COLORS["green"])
    add_arrow(ax, (0.85, 0.57), (0.85, 0.43), COLORS["purple"])
    add_arrow(ax, (0.75, 0.30), (0.68, 0.39), COLORS["blue"])
    ax.text(0.58, 0.18, "receding-horizon loop", ha="center", va="center", fontsize=8, color=COLORS["gray"])
    ax.text(0.58, 0.79, "one legal next node per LLM call", ha="center", va="center", fontsize=8, color=COLORS["green"])
    save(fig, "fig_stepwise_architecture")


def success_snapshot(rows: list[dict[str, str]]) -> None:
    labels = ["No LLM", "LFM2.5-1.2B", "Nemotron-4B", "LFM2.5-8B", "Gemma-3-1B", "Qwen3-1.7B"]
    models = ["no_llm", "liquid_lfm2.5-1.2b", "nvidia_nemotron-3-nano-4b", "lfm2.5-8b-a1b-mlx", "google_gemma-3-1b", "qwen_qwen3-1.7b"]
    long_vals = []
    sem_vals = []
    for model in models:
        mode = "no_llm" if model == "no_llm" else "llm_step"
        long_vals.append(f(find_row(rows, "long_horizon", mode, model) or {}, "success_rate") * 100)
        sem_vals.append(f(find_row(rows, "semantic_constraint", mode, model) or {}, "success_rate") * 100)
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.bar(x - width / 2, long_vals, width, color=COLORS["green"], label="Long horizon")
    ax.bar(x + width / 2, sem_vals, width, color=COLORS["orange"], label="Semantic constraint")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for xpos, vals in ((x - width / 2, long_vals), (x + width / 2, sem_vals)):
        for xx, val in zip(xpos, vals, strict=True):
            ax.text(xx, val + 1.7, f"{val:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_stepwise_success_snapshot")


def raw_vs_step(rows: list[dict[str, str]]) -> None:
    models = ["liquid_lfm2.5-1.2b", "nvidia_nemotron-3-nano-4b", "google_gemma-3-1b", "lfm2.5-8b-a1b-mlx", "qwen_qwen3-1.7b"]
    labels = [MODEL_LABELS[m] for m in models]
    raw_vals = [f(find_row(rows, "long_horizon", "llm_raw", m) or {}, "success_rate") * 100 for m in models]
    step_vals = [f(find_row(rows, "long_horizon", "llm_step", m) or {}, "success_rate") * 100 for m in models]
    strict_vals = [f(find_row(rows, "long_horizon", "llm_step", m) or {}, "strict_valid_rate") * 100 for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(x - 0.24, raw_vals, 0.24, color=COLORS["red"], label="llm_raw success")
    ax.bar(x, step_vals, 0.24, color=COLORS["green"], label="llm_step success")
    ax.bar(x + 0.24, strict_vals, 0.24, color=COLORS["blue"], label="llm_step strict valid")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_stepwise_raw_vs_step")


def semantic_tradeoff(rows: list[dict[str, str]]) -> None:
    models = ["no_llm", "nvidia_nemotron-3-nano-4b", "liquid_lfm2.5-1.2b", "lfm2.5-8b-a1b-mlx", "google_gemma-3-1b", "qwen_qwen3-1.7b"]
    labels = [MODEL_LABELS[m] for m in models]
    success = []
    cost = []
    for model in models:
        mode = "no_llm" if model == "no_llm" else "llm_step"
        row = find_row(rows, "semantic_constraint", mode, model) or {}
        success.append(f(row, "success_rate") * 100)
        cost.append(f(row, "mean_semantic_cost"))
    x = np.arange(len(models))
    fig, ax1 = plt.subplots(figsize=(7.2, 3.4))
    ax1.bar(x, success, 0.55, color=COLORS["green"], label="Success")
    ax1.set_ylim(0, 90)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    style_axis(ax1)
    ax2 = ax1.twinx()
    ax2.plot(x, cost, color=COLORS["purple"], marker="D", linewidth=1.8, label="Semantic cost")
    ax2.set_ylim(0, 0.17)
    ax2.set_ylabel("Mean semantic cost")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_stepwise_semantic_tradeoff")


def main() -> None:
    rows = load_rows()
    architecture()
    success_snapshot(rows)
    raw_vs_step(rows)
    semantic_tradeoff(rows)


if __name__ == "__main__":
    main()
