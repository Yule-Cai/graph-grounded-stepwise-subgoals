#!/usr/bin/env python3
"""Language-preference stress diagnostic for Paper A.

This diagnostic isolates a capability that graph search and fixed scalar
scorers do not naturally provide: mapping route-level natural-language
preferences to route choices when the preference is conflicting,
compositional, dynamic, or intentionally underspecified.

The script is pre-execution. It does not claim robot success. Closed-loop
execution remains handled by run_map_conditioned_llm_planning.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from urllib import error, request


DEFAULT_CASES = Path("paper_assets/paper_a/raw/preference_alignment_cases.csv")
DEFAULT_OUT = Path("paper_assets/paper_a/tables/language_preference_stress_results.csv")
DEFAULT_SUMMARY = Path("paper_assets/paper_a/tables/language_preference_stress_summary.csv")

OPTION_IDS = ["A", "B", "C", "D", "E"]


TASKS = [
    {
        "id": "shortest",
        "family": "basic",
        "texts": [
            "The delivery is urgent. Choose the feasible route with the smallest travel distance.",
            "Minimize distance. Risk and clearance are secondary unless the route is infeasible.",
        ],
        "target": "r_short",
    },
    {
        "id": "lowest_risk",
        "family": "basic",
        "texts": [
            "The robot carries fragile items. Prefer the route with the lowest semantic risk.",
            "Avoid semantic-risk regions as much as possible, even if the route is longer.",
        ],
        "target": "r_safe",
    },
    {
        "id": "conflicting_kitchen_distance",
        "family": "conflicting",
        "texts": [
            "Avoid the kitchen area unless going through it saves at least 30% travel distance. If the saving is smaller, take the best non-kitchen route.",
            "The kitchen is undesirable, but it is acceptable only for a very large shortcut of 30% or more. Otherwise avoid it.",
        ],
        "target": "r_balanced",
    },
    {
        "id": "compositional_quiet_dry",
        "family": "compositional",
        "texts": [
            "Prefer a quiet corridor, but do not use a wet-floor segment. If the quietest route is wet, choose the best dry alternative.",
            "Take a low-noise route while strictly avoiding wet floor. Balance quietness and dryness before distance.",
        ],
        "target": "r_balanced",
    },
    {
        "id": "multi_risk_priority",
        "family": "multi_risk",
        "texts": [
            "Wet floor is the most dangerous region, semantic risk is second, and noise is acceptable. Choose the route that best follows this priority.",
            "Avoid wet floor first, then semantic risk. A noisy hallway is acceptable if it keeps the robot away from wet and risky regions.",
        ],
        "target": "r_safe",
    },
    {
        "id": "dynamic_update_battery",
        "family": "dynamic",
        "texts": [
            "The initial preference was the safest route, but the battery is now low. Switch to the shortest route that still avoids severe wet-floor exposure.",
            "Update the route preference: conserve battery now, but do not enter wet-floor regions. Choose a short dry compromise.",
        ],
        "target": "r_balanced",
    },
    {
        "id": "disturbance_sensitive",
        "family": "hard_to_score",
        "texts": [
            "Avoid disturbing people: privacy-sensitive zones and narrow passages matter more than a brief kitchen crossing. Choose the least disruptive route.",
            "For a late-night delivery, avoid privacy-sensitive and narrow areas. Kitchen exposure is acceptable if brief.",
        ],
        "target": "r_smooth",
    },
    {
        "id": "ambiguous_safe_not_too_long",
        "family": "ambiguous",
        "texts": [
            "Choose the safer route, but only if it is not too long. The phrase 'too long' is not defined.",
            "Take a safe route unless the detour is excessive. No threshold for excessive detour is provided.",
        ],
        "target": "CLARIFY",
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


def route_options(case: dict[str, str], variant_idx: int) -> list[dict[str, object]]:
    """Construct five route options with controlled attribute trade-offs."""
    seed = f"{case.get('case_id', 'case')}::{variant_idx}"
    rng = random.Random(seed)
    base = max(6.0, float(case.get("euclidean_distance") or 10.0))
    jitter = lambda scale: rng.uniform(-scale, scale)
    return [
        {
            "route_id": "r_short",
            "description": "central shortcut through kitchen-adjacent corridor",
            "length": round(base * (1.00 + jitter(0.015)), 2),
            "turns": 2,
            "min_clearance": round(0.55 + jitter(0.03), 2),
            "semantic_risk": round(0.62 + jitter(0.025), 3),
            "kitchen_exposure": round(0.85 + jitter(0.02), 3),
            "wet_floor": round(0.35 + jitter(0.02), 3),
            "noise": round(0.20 + jitter(0.02), 3),
            "privacy_sensitive": round(0.30 + jitter(0.02), 3),
            "narrow_passage": round(0.40 + jitter(0.02), 3),
        },
        {
            "route_id": "r_safe",
            "description": "outer perimeter detour",
            "length": round(base * (1.32 + jitter(0.015)), 2),
            "turns": 5,
            "min_clearance": round(1.15 + jitter(0.03), 2),
            "semantic_risk": round(0.03 + jitter(0.01), 3),
            "kitchen_exposure": round(0.05 + jitter(0.01), 3),
            "wet_floor": round(0.02 + jitter(0.01), 3),
            "noise": round(0.45 + jitter(0.02), 3),
            "privacy_sensitive": round(0.12 + jitter(0.02), 3),
            "narrow_passage": round(0.15 + jitter(0.02), 3),
        },
        {
            "route_id": "r_smooth",
            "description": "sweeping staff corridor with very few turns",
            "length": round(base * (1.16 + jitter(0.015)), 2),
            "turns": 1,
            "min_clearance": round(0.82 + jitter(0.03), 2),
            "semantic_risk": round(0.14 + jitter(0.015), 3),
            "kitchen_exposure": round(0.30 + jitter(0.02), 3),
            "wet_floor": round(0.10 + jitter(0.015), 3),
            "noise": round(0.40 + jitter(0.02), 3),
            "privacy_sensitive": round(0.05 + jitter(0.015), 3),
            "narrow_passage": round(0.08 + jitter(0.015), 3),
        },
        {
            "route_id": "r_clear",
            "description": "wide quiet lobby detour with a wet-floor segment",
            "length": round(base * (1.24 + jitter(0.015)), 2),
            "turns": 4,
            "min_clearance": round(1.45 + jitter(0.03), 2),
            "semantic_risk": round(0.09 + jitter(0.015), 3),
            "kitchen_exposure": round(0.10 + jitter(0.015), 3),
            "wet_floor": round(0.55 + jitter(0.02), 3),
            "noise": round(0.05 + jitter(0.015), 3),
            "privacy_sensitive": round(0.60 + jitter(0.02), 3),
            "narrow_passage": round(0.02 + jitter(0.01), 3),
        },
        {
            "route_id": "r_balanced",
            "description": "moderate dry side corridor",
            "length": round(base * (1.11 + jitter(0.015)), 2),
            "turns": 3,
            "min_clearance": round(0.95 + jitter(0.03), 2),
            "semantic_risk": round(0.08 + jitter(0.015), 3),
            "kitchen_exposure": round(0.10 + jitter(0.015), 3),
            "wet_floor": round(0.06 + jitter(0.015), 3),
            "noise": round(0.16 + jitter(0.015), 3),
            "privacy_sensitive": round(0.10 + jitter(0.015), 3),
            "narrow_passage": round(0.12 + jitter(0.015), 3),
        },
    ]


def task_text(task: dict[str, object], variant_idx: int) -> str:
    texts = task["texts"]
    if not isinstance(texts, list):
        return str(texts)
    return str(texts[variant_idx % len(texts)])


def selected_tasks(benchmark: str) -> list[dict[str, object]]:
    if benchmark == "all":
        return TASKS
    if benchmark == "basic":
        return [task for task in TASKS if task["family"] == "basic"]
    if benchmark == "stress":
        return [task for task in TASKS if task["family"] != "basic"]
    raise ValueError(f"unknown benchmark={benchmark!r}")


def build_prompt(
    case: dict[str, str],
    task: dict[str, object],
    variant_idx: int,
    option_order: str,
) -> tuple[list[dict[str, str]], dict[str, str], str]:
    options = route_options(case, variant_idx)
    if option_order == "shuffled":
        rng = random.Random(f"{case['case_id']}::{task['id']}::{variant_idx}")
        rng.shuffle(options)
    elif option_order == "canonical":
        options = sorted(options, key=lambda item: str(item["route_id"]))
    else:
        raise ValueError(f"unknown option_order={option_order!r}")
    option_map: dict[str, str] = {}
    lines = [
        "Select one route option for a mobile robot, or return CLARIFY if the preference is underspecified.",
        f"Case: {case['case_id']}.",
        "All listed route options are graph-feasible. Lower risk/exposure values are better; higher min_clearance is better.",
        f"Preference: {task_text(task, variant_idx)}",
        "Options:",
    ]
    for option, item in zip(OPTION_IDS, options, strict=True):
        option_map[option] = str(item["route_id"])
        fields = ", ".join(
            f"{key}={item[key]}"
            for key in (
                "length",
                "turns",
                "min_clearance",
                "semantic_risk",
                "kitchen_exposure",
                "wet_floor",
                "noise",
                "privacy_sensitive",
                "narrow_passage",
            )
        )
        lines.append(f"{option}: {item['description']}; {fields}")
    lines.extend(
        [
            "Rules:",
            "- Choose exactly one option A-E when the preference is sufficiently specified.",
            "- Return CLARIFY when a key threshold is missing and two or more options could reasonably satisfy the preference.",
            "- Do not choose by option position; compare the route attributes and the language preference.",
            'Return JSON only, e.g. {"option":"B"} or {"option":"CLARIFY"}.',
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful route-preference interpreter for robot navigation. "
                "Map the natural-language preference to route attributes. Return only JSON."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ], option_map, "\n".join(lines)


def parse_option(text: str) -> str:
    cleaned = text.strip()
    json_match = re.search(r'"(?:option|answer|choice)"\s*:\s*"([A-E]|CLARIFY)"', cleaned, flags=re.I)
    if json_match:
        return json_match.group(1).upper()
    clarify = re.search(r"\b(CLARIFY|UNCLEAR|ASK)\b", cleaned, flags=re.I)
    if clarify:
        return "CLARIFY"
    match = re.search(r"\b([A-E])\b", cleaned)
    return match.group(1) if match else ""


def select_lmstudio(
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
    if option == "CLARIFY":
        return "CLARIFY", option, content
    return option_map.get(option, ""), option, content


def first_option(option_map: dict[str, str]) -> str:
    return option_map.get("A", "")


def hand_scorer(task_id: str, option_map: dict[str, str], options_by_id: dict[str, dict[str, object]]) -> str:
    """A deliberately simple deterministic scorer for comparison."""
    if task_id == "shortest":
        key = lambda rid: float(options_by_id[rid]["length"])
    elif task_id == "lowest_risk":
        key = lambda rid: float(options_by_id[rid]["semantic_risk"])
    elif task_id == "conflicting_kitchen_distance":
        shortest = min(options_by_id, key=lambda rid: float(options_by_id[rid]["length"]))
        shortest_len = float(options_by_id[shortest]["length"])
        non_kitchen = [rid for rid, row in options_by_id.items() if float(row["kitchen_exposure"]) <= 0.2]
        best_non_kitchen = min(non_kitchen, key=lambda rid: float(options_by_id[rid]["length"]))
        saving = (float(options_by_id[best_non_kitchen]["length"]) - shortest_len) / max(float(options_by_id[best_non_kitchen]["length"]), 1e-9)
        return shortest if saving >= 0.30 else best_non_kitchen
    elif task_id == "compositional_quiet_dry":
        key = lambda rid: 4.0 * float(options_by_id[rid]["wet_floor"]) + float(options_by_id[rid]["noise"])
    elif task_id == "multi_risk_priority":
        key = lambda rid: 8.0 * float(options_by_id[rid]["wet_floor"]) + 3.0 * float(options_by_id[rid]["semantic_risk"]) + 0.2 * float(options_by_id[rid]["noise"])
    elif task_id == "dynamic_update_battery":
        shortest_len = min(float(row["length"]) for row in options_by_id.values())
        feasible = [
            rid
            for rid, row in options_by_id.items()
            if float(row["wet_floor"]) <= 0.12 and float(row["length"]) <= 1.15 * shortest_len
        ]
        return min(feasible or list(options_by_id), key=lambda rid: float(options_by_id[rid]["length"]))
    elif task_id == "disturbance_sensitive":
        key = lambda rid: 4.0 * float(options_by_id[rid]["privacy_sensitive"]) + 2.0 * float(options_by_id[rid]["narrow_passage"]) + 0.5 * float(options_by_id[rid]["kitchen_exposure"])
    elif task_id == "ambiguous_safe_not_too_long":
        return "CLARIFY"
    else:
        key = lambda rid: float(options_by_id[rid]["length"])
    return min(options_by_id, key=key)


def mean_int(rows: list[dict[str, object]], key: str) -> float:
    return sum(int(row[key]) for row in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model"]) == model]
        for group_name, group_rows in [("overall", model_rows)]:
            out.append(
                {
                    "model": model,
                    "group": group_name,
                    "family": "all",
                    "task_id": "all",
                    "rows": len(group_rows),
                    "parse_ok_rate": f"{mean_int(group_rows, 'parse_ok'):.4f}",
                    "match_rate": f"{mean_int(group_rows, 'match'):.4f}",
                    "clarify_target_rate": f"{mean_int(group_rows, 'clarify_target'):.4f}",
                    "clarify_correct_rate": f"{mean_int(group_rows, 'clarify_correct'):.4f}",
                    "first_option_match_rate": f"{mean_int(group_rows, 'first_option_match'):.4f}",
                    "hand_scorer_match_rate": f"{mean_int(group_rows, 'hand_scorer_match'):.4f}",
                }
            )
        for family in sorted({str(row["family"]) for row in model_rows}):
            family_rows = [row for row in model_rows if str(row["family"]) == family]
            out.append(
                {
                    "model": model,
                    "group": "by_family",
                    "family": family,
                    "task_id": "all",
                    "rows": len(family_rows),
                    "parse_ok_rate": f"{mean_int(family_rows, 'parse_ok'):.4f}",
                    "match_rate": f"{mean_int(family_rows, 'match'):.4f}",
                    "clarify_target_rate": f"{mean_int(family_rows, 'clarify_target'):.4f}",
                    "clarify_correct_rate": f"{mean_int(family_rows, 'clarify_correct'):.4f}",
                    "first_option_match_rate": f"{mean_int(family_rows, 'first_option_match'):.4f}",
                    "hand_scorer_match_rate": f"{mean_int(family_rows, 'hand_scorer_match'):.4f}",
                }
            )
        for task_id in sorted({str(row["task_id"]) for row in model_rows}):
            task_rows = [row for row in model_rows if str(row["task_id"]) == task_id]
            out.append(
                {
                    "model": model,
                    "group": "by_task",
                    "family": str(task_rows[0]["family"]),
                    "task_id": task_id,
                    "rows": len(task_rows),
                    "parse_ok_rate": f"{mean_int(task_rows, 'parse_ok'):.4f}",
                    "match_rate": f"{mean_int(task_rows, 'match'):.4f}",
                    "clarify_target_rate": f"{mean_int(task_rows, 'clarify_target'):.4f}",
                    "clarify_correct_rate": f"{mean_int(task_rows, 'clarify_correct'):.4f}",
                    "first_option_match_rate": f"{mean_int(task_rows, 'first_option_match'):.4f}",
                    "hand_scorer_match_rate": f"{mean_int(task_rows, 'hand_scorer_match'):.4f}",
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate language-conditioned route-preference stress cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--benchmark", choices=("basic", "stress", "all"), default="all")
    parser.add_argument("--lm-studio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--variants-per-task", type=int, default=2)
    parser.add_argument("--option-order", choices=("shuffled", "canonical"), default="shuffled")
    parser.add_argument("--write-prompts-only", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases, args.limit)
    tasks = selected_tasks(args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case in cases:
        for task in tasks:
            for variant_idx in range(args.variants_per_task):
                messages, option_map, prompt_text = build_prompt(case, task, variant_idx, args.option_order)
                options = route_options(case, variant_idx)
                options_by_id = {str(item["route_id"]): item for item in options}
                target = str(task["target"])
                selected = ""
                selected_option = ""
                raw = ""
                error_text = ""
                if not args.write_prompts_only:
                    try:
                        selected, selected_option, raw = select_lmstudio(
                            args.lm_studio_url,
                            args.model,
                            messages,
                            args.timeout_s,
                            args.max_tokens,
                            option_map,
                        )
                    except (OSError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                        error_text = str(exc)
                first = first_option(option_map)
                scored = hand_scorer(str(task["id"]), option_map, options_by_id)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "task_id": task["id"],
                        "family": task["family"],
                        "variant": variant_idx,
                        "preference_text": task_text(task, variant_idx),
                        "target_route_id": target,
                        "target_option": next((option for option, route_id in option_map.items() if route_id == target), target),
                        "selected_route_id": selected,
                        "selected_option": selected_option,
                        "first_option_route_id": first,
                        "hand_scorer_route_id": scored,
                        "first_option_match": int(first == target),
                        "hand_scorer_match": int(scored == target),
                        "match": int(bool(selected) and selected == target),
                        "parse_ok": int(bool(selected)),
                        "clarify_target": int(target == "CLARIFY"),
                        "clarify_correct": int(target == "CLARIFY" and selected == "CLARIFY"),
                        "option_order": args.option_order,
                        "model": args.model,
                        "prompt": json.dumps(messages, ensure_ascii=False),
                        "prompt_text": prompt_text,
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

    overall = summary[0]
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote summary to {args.summary_output}")
    print(
        "match_rate={match_rate} parse_ok_rate={parse_ok_rate} "
        "hand_scorer_match_rate={hand_scorer_match_rate} first_option_match_rate={first_option_match_rate}".format(**overall)
    )
    if args.write_prompts_only:
        print("prompts only: no LM Studio calls were made")


if __name__ == "__main__":
    main()
