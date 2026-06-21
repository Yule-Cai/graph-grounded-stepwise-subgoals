#!/usr/bin/env python3
"""Generate compact Paper A figures designed for AAAI single-column placement."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("paper_assets/paper_a/figures")


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_results() -> None:
    methods = ["Direct", "Random WP", "Classical WP", "Fast-safe", "Rule sem.", "Valid. LLM"]
    long_horizon = np.array([43.0, 76.0, 81.0, 64.0, np.nan, 81.0])
    semantic = np.array([42.0, 69.0, 89.0, 59.0, 61.0, 78.0])

    y = np.arange(len(methods))
    height = 0.34
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.barh(y + height / 2, long_horizon, height=height, color="#4f8fc0", label="Long")
    ax.barh(y - height / 2, semantic, height=height, color="#2f9e44", label="Semantic")
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Success rate (%)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    for values, offset in ((long_horizon, height / 2), (semantic, -height / 2)):
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(value + 1.2, i + offset, f"{value:.0f}", va="center", fontsize=6.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig1_main_results_compact")


def llm_validity() -> None:
    settings = ["Long", "Semantic"]
    parse = np.array([79.0, 99.0])
    strict = np.array([0.0, 0.0])
    repair = np.array([100.0, 100.0])
    success = np.array([81.0, 78.0])

    x = np.arange(len(settings))
    width = 0.19
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    ax.bar(x - 1.5 * width, parse, width=width, color="#4f8fc0", label="Parse OK")
    ax.bar(x - 0.5 * width, strict, width=width, color="#2f9e44", label="Strict")
    ax.bar(x + 0.5 * width, repair, width=width, color="#d98c2b", label="Repaired")
    ax.bar(x + 1.5 * width, success, width=width, color="#775aa6", label="Success")
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=7.3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Rate (%)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6.2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig4_llm_validity_compact")


if __name__ == "__main__":
    main_results()
    llm_validity()
