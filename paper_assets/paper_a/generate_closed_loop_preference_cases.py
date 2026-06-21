#!/usr/bin/env python3
"""Generate closed-loop route-preference cases for Paper A.

The natural-language preference diagnostic is pre-execution. This generator
turns the same semantic start-goal-risk cases into closed-loop navigation cases
with different route-level preference texts, so the existing evaluator can test
whether preference-conditioned route choices survive downstream execution.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("paper_assets/paper_a/raw/preference_alignment_cases.csv")

BASIC_PREFERENCES = [
    (
        "shortest",
        "The delivery is urgent. Prioritize the shortest feasible route while still using legal graph transitions.",
    ),
    (
        "lowest_risk",
        "The marked red region is temporarily unsafe. Prefer a safer route that avoids the risk region, even if longer.",
    ),
    (
        "fewest_turns",
        "The robot is carrying an open cup. Prefer a smoother route with fewer high-level turns.",
    ),
    (
        "high_clearance",
        "The robot is carrying a wide tray. Prefer wider, lower-risk passages over narrow shortcuts.",
    ),
    (
        "balanced",
        "Prefer a practical compromise: avoid obvious risk, but do not take an excessive detour.",
    ),
]

STRESS_PREFERENCES = [
    (
        "conflicting_kitchen_distance",
        "Avoid kitchen-like semantic-risk regions unless the route saves a large amount of travel distance; if the saving is modest, choose a safer non-kitchen detour.",
    ),
    (
        "compositional_quiet_dry",
        "Prefer the quieter side corridor, but strictly avoid wet-floor or unsafe semantic-risk regions; balance quietness and dryness before distance.",
    ),
    (
        "multi_risk_priority",
        "Treat wet-floor risk as most important, ordinary semantic risk as second, and extra distance as less important; choose the route that respects this priority.",
    ),
    (
        "dynamic_update_battery",
        "The original goal was maximum safety, but the battery is now low; switch to the shortest route that still avoids severe semantic-risk exposure.",
    ),
    (
        "disturbance_sensitive",
        "Avoid disturbing people: privacy-sensitive and narrow areas matter more than a brief passage near ordinary service spaces; choose the least disruptive route.",
    ),
]


def preference_rows(preference_set: str) -> list[tuple[str, str]]:
    if preference_set == "basic":
        return BASIC_PREFERENCES
    if preference_set == "stress":
        return STRESS_PREFERENCES
    if preference_set == "all":
        return BASIC_PREFERENCES + STRESS_PREFERENCES
    raise ValueError(f"unknown preference_set={preference_set!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-limit", type=int, default=20)
    parser.add_argument("--map-id", default="reference_family_flat")
    parser.add_argument("--preference-set", choices=("basic", "stress", "all"), default="basic")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        base_rows = list(csv.DictReader(handle))[: args.base_limit]
    if not base_rows:
        raise SystemExit(f"no cases found in {args.input}")

    out_rows: list[dict[str, str]] = []
    prefs = preference_rows(args.preference_set)
    for base in base_rows:
        for pref_id, pref_text in prefs:
            row = dict(base)
            row["case_id"] = f"{pref_id}_{base['case_id']}"
            row["scenario"] = "semantic_constraint"
            row["map_id"] = row.get("map_id") or args.map_id
            row["difficulty_tag"] = f"closed_loop_preference_{pref_id}"
            row["route_constraint"] = pref_text
            row["preference_id"] = pref_id
            row["preference_text"] = pref_text
            out_rows.append(row)

    fieldnames = list(base_rows[0].keys())
    for extra in ("preference_id", "preference_text"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} closed-loop preference cases to {args.output} preference_set={args.preference_set}")


if __name__ == "__main__":
    main()
