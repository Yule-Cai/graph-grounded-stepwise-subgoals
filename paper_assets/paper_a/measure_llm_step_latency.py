#!/usr/bin/env python3
"""Measure LM Studio latency for Paper A stepwise next-subgoal prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
import time
from pathlib import Path
from typing import Any

from llm_rl_nav.training.eval_paper_a_cases import _apply_case_reset, _load_cases, _make_env
from llm_rl_nav.training.eval_llm_route_planning import _base_env
from llm_rl_nav.training.train_multimap_ppo import parse_maps

from run_map_conditioned_llm_planning import (
    _build_compact_graph,
    _candidate_features,
    _chat_completion_data,
    _hop_distances_to_goal,
    _parse_llm_next_node,
)


def make_prompt(graph: dict[str, Any], case: dict[str, Any], args) -> tuple[dict[str, Any], set[str], int]:
    current = "S"
    coords = graph["coords"]
    goal_xy = coords["G"]
    current_xy = coords[current]
    allowed = [node for node in graph["adjacency"].get(current, []) if node == "G" or node != "S"]
    current_goal_distance = math.hypot(current_xy[0] - goal_xy[0], current_xy[1] - goal_xy[1])
    hop_distances = _hop_distances_to_goal(graph)
    rows = [_candidate_features(graph, current, node, current_goal_distance, hop_distances) for node in allowed]
    rows.sort(key=lambda item: (not item["is_goal"], item["hops_to_goal"], -item["progress_delta"], item["risk_score"]))
    payload = {
        "map_source": "ROS2/Gazebo 3D semantic world projected to a compact route graph",
        "case_id": case.get("case_id"),
        "scenario": graph["scenario"],
        "task": case.get("route_constraint"),
        "current_node": current,
        "current_distance_to_goal": round(current_goal_distance, 3),
        "step_index": 0,
        "goal_node": "G",
        "visited_nodes": ["S"],
        "allowed_next_nodes": rows,
        "risk_zone": {
            "center": [round(graph["risk_center"][0], 2), round(graph["risk_center"][1], 2)],
            "radius": round(graph["risk_radius"], 2),
            "meaning": "semantic/risk/narrow region; avoid when the task asks for safer routing",
        },
        "instruction": (
            "Choose exactly one next subgoal node from allowed_next_nodes. "
            "Candidate order is a presentation detail, not a route ranking; compare the numeric fields. "
            "Return one legal next node only."
        ),
    }
    prompt = {
        "model": args.lm_model,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are an edge-deployed high-level subgoal planner for a mobile robot.",
                        "At each step, choose one legal next node from the allowed_next_nodes list.",
                        "Return JSON only with this schema: {\"next_node\":\"<node_id>\",\"reason\":\"short\"}.",
                        "If G is allowed, output G immediately.",
                        "Do not use candidate list order as the main criterion; use the candidate features.",
                    ]
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.llm_step_max_tokens,
    }
    return prompt, set(allowed), len(rows)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo", choices=("ppo", "sac", "a2c", "dqn"))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--maps", type=parse_maps, default=("reference_family_flat",))
    parser.add_argument("--map-source", default="gazebo_3d_projection")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--reward-profile", default="v8_goal")
    parser.add_argument("--goal-min-distance", type=float, default=2.0)
    parser.add_argument("--goal-max-distance", type=float, default=12.0)
    parser.add_argument("--goal-point-probability", type=float, default=0.95)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--shield-min-clearance", type=float, default=0.18)
    parser.add_argument("--shield-intervention-penalty", type=float, default=0.25)
    parser.add_argument("--lm-studio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--lm-model", required=True)
    parser.add_argument("--llm-timeout-s", type=float, default=60.0)
    parser.add_argument("--llm-step-max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--graph-resolution", type=float, default=2.2)
    parser.add_argument("--waypoint-spacing", type=float, default=2.2)
    parser.add_argument("--waypoint-grid-resolution", type=float, default=0.55)
    parser.add_argument("--graph-margin", type=float, default=4.5)
    parser.add_argument("--edge-radius", type=float, default=3.5)
    parser.add_argument("--max-graph-nodes", type=int, default=42)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--semantic-cost-weight", type=float, default=5.0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    cases = _load_cases(args.cases)[: args.samples]
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        env = _make_env(args, str(case.get("map_id") or args.maps[0]), args.seed + index)
        _obs, _start, _goal, _map_id = _apply_case_reset(env, case, args.seed + index)
        graph = _build_compact_graph(_base_env(env), case, args)
        prompt, allowed, candidate_count = make_prompt(graph, case, args)
        prompt_chars = sum(len(str(message.get("content", ""))) for message in prompt["messages"])
        t0 = time.perf_counter()
        ok = 0
        node = None
        error = ""
        content_chars = 0
        try:
            data = _chat_completion_data(prompt, args)
            latency = time.perf_counter() - t0
            node, meta, _raw = _parse_llm_next_node(data, allowed, True)
            ok = int(bool(node))
            content_chars = int(meta.get("content_chars", 0) or 0) + int(meta.get("reasoning_chars", 0) or 0)
        except Exception as exc:
            latency = time.perf_counter() - t0
            error = str(exc)[:240]
        rows.append(
            {
                "model": args.lm_model,
                "case_id": case.get("case_id", ""),
                "scenario": case.get("scenario", ""),
                "map_id": case.get("map_id", ""),
                "candidate_count": candidate_count,
                "prompt_chars": prompt_chars,
                "output_chars": content_chars,
                "latency_s": latency,
                "parse_ok": ok,
                "selected_node": node or "",
                "error": error,
            }
        )
        print(
            f"{index+1:03d}/{len(cases):03d} model={args.lm_model} case={case.get('case_id')} "
            f"latency={latency:.3f}s parse_ok={ok} node={node or '-'}"
        )

    write_rows(args.out_csv, rows)
    latencies = [float(row["latency_s"]) for row in rows]
    parsed = sum(int(row["parse_ok"]) for row in rows)
    summary = [
        {
            "model": args.lm_model,
            "samples": len(rows),
            "parse_ok_rate": parsed / max(len(rows), 1),
            "mean_latency_s": stats.mean(latencies) if latencies else 0.0,
            "median_latency_s": stats.median(latencies) if latencies else 0.0,
            "p95_latency_s": percentile(latencies, 95),
            "min_latency_s": min(latencies) if latencies else 0.0,
            "max_latency_s": max(latencies) if latencies else 0.0,
            "mean_prompt_chars": stats.mean([int(row["prompt_chars"]) for row in rows]) if rows else 0.0,
            "mean_output_chars": stats.mean([int(row["output_chars"]) for row in rows]) if rows else 0.0,
        }
    ]
    write_rows(args.summary_csv, summary)
    print(f"latency_csv: {args.out_csv}")
    print(f"summary_csv: {args.summary_csv}")


if __name__ == "__main__":
    main()
