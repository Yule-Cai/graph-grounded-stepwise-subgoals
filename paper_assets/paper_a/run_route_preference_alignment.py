#!/usr/bin/env python3
"""Route-preference alignment diagnostic for Paper A.

This script evaluates whether a local LLM can map route-level natural-language
preferences to candidate route ids. It is intentionally pre-execution: the
validator and robot rollout are separate from this language-alignment check.
The diagnostic is useful for reviewers because it measures a setting where the
route objective is stated in language rather than fully hand-coded as one scalar
edge cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
from urllib import error, request


DEFAULT_CASES = Path("paper_assets/paper_a/raw/preference_alignment_cases.csv")
DEFAULT_OUT = Path("paper_assets/paper_a/tables/route_preference_alignment_results.csv")
DEFAULT_SUMMARY = Path("paper_assets/paper_a/tables/route_preference_alignment_summary.csv")

PREFERENCES = [
    {
        "id": "shortest",
        "texts": [
            "The delivery is urgent and all listed routes are feasible. Prioritize the smallest travel distance.",
            "Choose the shortest route; length is the primary criterion in this diagnostic case.",
            "Minimize travel distance. Treat risk and clearance as secondary unless the route is infeasible.",
        ],
        "target": "r_short",
    },
    {
        "id": "lowest_risk",
        "texts": [
            "The marked red region is temporarily unsafe. Prefer the route that stays farthest from it, even if longer.",
            "Avoid the semantic-risk region as much as possible; a longer detour is acceptable.",
            "The robot is carrying fragile items. Select the safest route with the lowest risk exposure.",
        ],
        "target": "r_safe",
    },
    {
        "id": "fewest_turns",
        "texts": [
            "The robot is carrying an open cup. Prioritize the route with the fewest turns.",
            "Choose the smoothest route; the number of turns is the primary criterion.",
            "Avoid zig-zag motion. Select the route with the smallest turn count.",
        ],
        "target": "r_smooth",
    },
    {
        "id": "high_clearance",
        "texts": [
            "The robot is carrying a wide tray. Prioritize the route with the highest clearance from obstacles and narrow passages.",
            "Avoid narrow passages; choose the route with the largest min_clearance value.",
            "The payload may scrape nearby furniture, so clearance is the primary criterion.",
        ],
        "target": "r_clear",
    },
    {
        "id": "balanced",
        "texts": [
            "Choose a balanced route that avoids high risk without being much longer.",
            "Do not take the risky shortcut, but also avoid an excessive detour.",
            "Prefer a practical compromise between short travel distance and low risk exposure.",
        ],
        "target": "r_balanced",
    },
]


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def load_cases(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def route_candidates(case: dict[str, str]) -> list[dict[str, object]]:
    sx, sy = float(case["start_x"]), float(case["start_y"])
    gx, gy = float(case["goal_x"]), float(case["goal_y"])
    rx, ry = float(case["risk_center_x"]), float(case["risk_center_y"])
    direct = math.hypot(gx - sx, gy - sy)
    risk_dist = max(0.2, distance_point_to_segment((rx, ry), (sx, sy), (gx, gy)))
    return [
        {
            "route_id": "r_short",
            "description": "central direct corridor",
            "length": round(direct, 2),
            "turns": 2,
            "min_clearance": 0.55,
            "semantic_risk": round(1.0 / risk_dist, 3),
        },
        {
            "route_id": "r_safe",
            "description": "outer perimeter detour",
            "length": round(direct * 1.32, 2),
            "turns": 5,
            "min_clearance": 1.15,
            "semantic_risk": 0.03,
        },
        {
            "route_id": "r_smooth",
            "description": "long sweeping corridor",
            "length": round(direct * 1.16, 2),
            "turns": 1,
            "min_clearance": 0.82,
            "semantic_risk": 0.14,
        },
        {
            "route_id": "r_clear",
            "description": "wide lobby detour",
            "length": round(direct * 1.24, 2),
            "turns": 4,
            "min_clearance": 1.45,
            "semantic_risk": 0.09,
        },
        {
            "route_id": "r_balanced",
            "description": "moderate side corridor",
            "length": round(direct * 1.10, 2),
            "turns": 3,
            "min_clearance": 0.95,
            "semantic_risk": 0.08,
        },
    ]


def distance_point_to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def preference_text(preference: dict[str, object], variant_idx: int) -> str:
    texts = preference["texts"]
    if not isinstance(texts, list):
        return str(texts)
    return str(texts[variant_idx % len(texts)])


def build_prompt(
    case: dict[str, str],
    preference: dict[str, object],
    variant_idx: int,
    option_order: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    candidates = route_candidates(case)
    if option_order == "shuffled":
        rng = random.Random(f"{case['case_id']}::{preference['id']}::{variant_idx}")
        rng.shuffle(candidates)
    elif option_order == "canonical":
        candidates = sorted(candidates, key=lambda item: str(item["route_id"]))
    else:
        raise ValueError(f"unknown option_order={option_order!r}")
    option_map: dict[str, str] = {}
    pref_text = preference_text(preference, variant_idx)
    lines = [
        "Select one route option for a mobile robot.",
        f"Case: {case['case_id']}.",
        "Map note: red annotated regions are semantic-risk regions. Lower semantic_risk is safer; higher min_clearance means wider free space.",
        f"Preference: {pref_text}",
        "Important: follow the stated preference exactly. Do not always choose the safest option unless the preference prioritizes risk avoidance.",
        "Options:",
    ]
    for option, item in zip(["A", "B", "C", "D", "E"], candidates, strict=True):
        item = {**item, "option": option}
        option_map[option] = str(item["route_id"])
        lines.append(
            "{option}: {description}; length={length}; "
            "turns={turns}; min_clearance={min_clearance}; semantic_risk={semantic_risk}".format(**item)
        )
    lines.append("Choose the option that best matches the preference.")
    lines.append(
        "Decision guide: shortest means smallest length; lowest risk means smallest semantic_risk; "
        "fewest turns means smallest turns; highest clearance means largest min_clearance; "
        "balanced means low semantic_risk without an extreme length increase."
    )
    lines.append('Return JSON only, for example {"option":"A"}.')
    return [
        {
            "role": "system",
            "content": (
                "You are a careful route-preference evaluator. Compare the route attributes numerically, "
                "follow the user's preference exactly, and return only JSON with one key named option."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ], option_map


def select_route_lmstudio(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float,
    max_tokens: int,
    option_map: dict[str, str],
) -> tuple[str, str, str]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LM_STUDIO_API_KEY") or "lm-studio"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in str(base_url):
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost/paper-a")
        title = os.environ.get("OPENROUTER_X_TITLE", "Paper A Robot Navigation Diagnostics")
        headers.update({"HTTP-Referer": referer, "X-Title": title})
    req = request.Request(
        chat_url(base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    option = parse_option(content)
    return option_map.get(option, ""), option, content


def parse_option(text: str) -> str:
    cleaned = text.strip()
    json_match = re.search(r'"(?:option|answer)"\s*:\s*"([A-E])"', cleaned)
    if json_match:
        return json_match.group(1)
    match = re.search(r"\b([A-E])\b", cleaned)
    return match.group(1) if match else ""


def mean_int(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(int(row[key]) for row in rows) / len(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    if not rows:
        return summary
    model = str(rows[0]["model"])
    summary.append(
        {
            "model": model,
            "group": "overall",
            "preference_id": "all",
            "rows": len(rows),
            "parse_ok_rate": f"{mean_int(rows, 'parse_ok'):.4f}",
            "match_rate": f"{mean_int(rows, 'match'):.4f}",
            "first_option_match_rate": f"{mean_int(rows, 'first_option_match'):.4f}",
            "random_expected_rate": "0.2000",
        }
    )
    pref_ids = sorted({str(row["preference_id"]) for row in rows})
    for pref_id in pref_ids:
        pref_rows = [row for row in rows if row["preference_id"] == pref_id]
        summary.append(
            {
                "model": model,
                "group": "by_preference",
                "preference_id": pref_id,
                "rows": len(pref_rows),
                "parse_ok_rate": f"{mean_int(pref_rows, 'parse_ok'):.4f}",
                "match_rate": f"{mean_int(pref_rows, 'match'):.4f}",
                "first_option_match_rate": f"{mean_int(pref_rows, 'first_option_match'):.4f}",
                "random_expected_rate": "0.2000",
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate route-level preference alignment.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--lm-studio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--variants-per-preference", type=int, default=2)
    parser.add_argument("--option-order", choices=["shuffled", "canonical"], default="shuffled")
    parser.add_argument("--write-prompts-only", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case in cases:
        for pref in PREFERENCES:
            for variant_idx in range(args.variants_per_preference):
                messages, option_map = build_prompt(case, pref, variant_idx, args.option_order)
                selected = ""
                selected_option = ""
                raw = ""
                error_text = ""
                if not args.write_prompts_only:
                    try:
                        selected, selected_option, raw = select_route_lmstudio(
                            args.lm_studio_url,
                            args.model,
                            messages,
                            args.timeout_s,
                            args.max_tokens,
                            option_map,
                        )
                    except (OSError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                        error_text = str(exc)
                target = str(pref["target"])
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "preference_id": pref["id"],
                        "preference_variant": variant_idx,
                        "preference_text": preference_text(pref, variant_idx),
                        "target_route_id": target,
                        "target_option": next((option for option, route_id in option_map.items() if route_id == target), ""),
                        "selected_route_id": selected,
                        "selected_option": selected_option,
                        "first_option_route_id": option_map.get("A", ""),
                        "first_option_match": int(option_map.get("A", "") == target),
                        "match": int(bool(selected) and selected == target),
                        "parse_ok": int(bool(selected)),
                        "option_order": args.option_order,
                        "model": args.model,
                        "prompt": json.dumps(messages, ensure_ascii=False),
                        "raw_response": raw,
                        "error": error_text,
                    }
                )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    if not any(row["selected_route_id"] for row in rows):
        print(f"wrote {len(rows)} prompt rows to {args.output}")
        print("No model responses were collected. Start LM Studio or use --write-prompts-only intentionally.")
        return
    overall = summary[0]
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote summary to {args.summary_output}")
    print(
        "preference_match_rate={match_rate} parse_ok_rate={parse_ok_rate} "
        "first_option_match_rate={first_option_match_rate}".format(**overall)
    )


if __name__ == "__main__":
    main()
