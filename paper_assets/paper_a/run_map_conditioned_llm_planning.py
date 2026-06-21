#!/usr/bin/env python3
"""Map-conditioned LLM waypoint planning for Paper A.

This evaluator differs from the earlier proactive route-option experiment:
the prompt contains a compact graph derived from the Gazebo semantic-world
projection, not a list of pre-scored route options. The LLM must output a
waypoint/node sequence, which is then validated, optionally graph-repaired,
and executed by the frozen PPO/SAC controller.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any
from urllib import error, request

from llm_rl_nav.training.eval_llm_route_planning import (
    _base_env,
    _candidate_route,
    _chat_url,
    _current_obs,
    _json_from_text,
    _predict_action,
    _sanitize_route,
    _set_goal,
)
from llm_rl_nav.training.eval_paper_a_cases import (
    _apply_case_reset,
    _float,
    _load_cases,
    _make_env,
)
from llm_rl_nav.training.eval_paper_a_proactive_route import (
    _base_clearance,
    _route_length,
    _semantic_cost,
    _trajectory_length,
    _trajectory_semantic_cost,
    _turn_count,
)
from llm_rl_nav.training.train_multimap_ppo import parse_maps


_HF_CHAT_CACHE: dict[str, Any] = {}
LLM_STEP_MODES = {"llm_step", "llm_step_retry", "llm_step_order_ensemble", "llm_step_consistency_gate"}


def _prompt_text_for_local_hf(prompt: dict[str, Any]) -> str:
    parts = []
    for message in prompt.get("messages", []):
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", ""))
        parts.append(f"{role}:\n{content}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def _local_hf_chat_completion(prompt: dict[str, Any], args) -> dict[str, Any]:
    """Run a local Hugging Face causal LM when --lm-model is a model directory."""
    model_dir = str(args.lm_model)
    cached = _HF_CHAT_CACHE.get(model_dir)
    if cached is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Local HF model execution requires torch and transformers in the active environment. "
                "Use LM Studio model ids instead, or run through ros_env."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype="auto")
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")
        model.eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        cached = (tokenizer, model)
        _HF_CHAT_CACHE[model_dir] = cached
    tokenizer, model = cached

    messages = prompt.get("messages", [])
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = _prompt_text_for_local_hf(prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    max_new_tokens = int(prompt.get("max_tokens", args.llm_max_tokens))
    temperature = float(prompt.get("temperature", args.temperature))
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": max(temperature, 1e-4)})
    else:
        generation_kwargs.update({"do_sample": False})
    with getattr(__import__("torch"), "no_grad")():
        output = model.generate(**inputs, **generation_kwargs)
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"choices": [{"message": {"content": text}}]}


def _chat_completion_data(prompt: dict[str, Any], args) -> dict[str, Any]:
    if Path(str(args.lm_model)).is_dir():
        return _local_hf_chat_completion(prompt, args)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LM_STUDIO_API_KEY") or "lm-studio"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in str(args.lm_studio_url):
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost/paper-a")
        title = os.environ.get("OPENROUTER_X_TITLE", "Paper A Robot Navigation Diagnostics")
        headers.update({"HTTP-Referer": referer, "X-Title": title})
    req = request.Request(
        _chat_url(args.lm_studio_url),
        data=json.dumps(prompt, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with request.urlopen(req, timeout=args.llm_timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc


def _line_free(base, a: tuple[float, float], b: tuple[float, float], step: float = 0.35) -> bool:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(math.ceil(dist / step)))
    for i in range(n + 1):
        t = i / n
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        if base._in_collision(x, y):
            return False
    return True


def _risk_value(point: tuple[float, float], center: tuple[float, float], radius: float) -> float:
    if radius <= 0:
        return 0.0
    dist = math.hypot(point[0] - center[0], point[1] - center[1])
    return max(0.0, (radius - dist) / max(radius, 1e-6))


def _graph_score(
    path: list[str],
    coords: dict[str, tuple[float, float]],
    start: tuple[float, float],
    risk_center: tuple[float, float],
    risk_radius: float,
    scenario: str,
) -> float:
    route = [coords[node] for node in path[1:] if node in coords]
    length = _route_length(route, start)
    turns = _turn_count([start] + route)
    semantic = _semantic_cost(route, risk_center, risk_radius) if scenario == "semantic_constraint" else 0.0
    return length + 1.8 * turns + 0.35 * len(route) + 5.0 * semantic


def _build_compact_graph(base, case: dict[str, Any], args) -> dict[str, Any]:
    start = (_float(case, "start_x"), _float(case, "start_y"))
    goal = (_float(case, "goal_x"), _float(case, "goal_y"))
    risk_center = (_float(case, "risk_center_x"), _float(case, "risk_center_y"))
    risk_radius = _float(case, "risk_radius", 0.0)
    scenario = str(case.get("scenario") or "long_horizon")
    perturb_rng = random.Random(f"{args.seed}:{case.get('case_id', '')}:graph_perturb")
    if scenario == "semantic_constraint" and args.risk_center_noise > 0:
        risk_center = (
            risk_center[0] + perturb_rng.uniform(-args.risk_center_noise, args.risk_center_noise),
            risk_center[1] + perturb_rng.uniform(-args.risk_center_noise, args.risk_center_noise),
        )
    if scenario == "semantic_constraint" and args.risk_radius_scale > 0:
        risk_radius *= args.risk_radius_scale

    min_x = min(start[0], goal[0], risk_center[0]) - args.graph_margin
    max_x = max(start[0], goal[0], risk_center[0]) + args.graph_margin
    min_y = min(start[1], goal[1], risk_center[1]) - args.graph_margin
    max_y = max(start[1], goal[1], risk_center[1]) + args.graph_margin
    if hasattr(base, "x_bounds"):
        min_x = max(min_x, float(base.x_bounds[0]))
        max_x = min(max_x, float(base.x_bounds[1]))
    if hasattr(base, "y_bounds"):
        min_y = max(min_y, float(base.y_bounds[0]))
        max_y = min(max_y, float(base.y_bounds[1]))

    # Seed the graph with a connected navigation skeleton so the prompt is a
    # map-to-route problem rather than an impossible graph puzzle. The LLM still
    # receives nodes and edges, not pre-scored route options or route ids.
    skeleton = _candidate_route(base, goal, spacing=args.waypoint_spacing, resolution=args.waypoint_grid_resolution)
    raw_points: list[tuple[float, float]] = [start, goal, *skeleton]
    if scenario == "semantic_constraint":
        raw_points.append(risk_center)

    x = math.floor(min_x / args.graph_resolution) * args.graph_resolution
    while x <= max_x + 1e-6:
        y = math.floor(min_y / args.graph_resolution) * args.graph_resolution
        while y <= max_y + 1e-6:
            point = (round(x, 2), round(y, 2))
            if not base._in_collision(point[0], point[1]):
                raw_points.append(point)
            y += args.graph_resolution
        x += args.graph_resolution

    # Keep the graph prompt compact: prioritize start-goal corridor, risk boundary,
    # and higher-clearance points.
    sx, sy = start
    gx, gy = goal
    line_len = math.hypot(gx - sx, gy - sy) or 1.0

    def priority(point: tuple[float, float]) -> tuple[float, float]:
        px, py = point
        t = max(0.0, min(1.0, ((px - sx) * (gx - sx) + (py - sy) * (gy - sy)) / (line_len * line_len)))
        proj = (sx + t * (gx - sx), sy + t * (gy - sy))
        corridor = math.hypot(px - proj[0], py - proj[1])
        risk_ring = abs(math.hypot(px - risk_center[0], py - risk_center[1]) - risk_radius) if risk_radius > 0 else 99.0
        clearance_bonus = -min(_base_clearance(base, px, py), 2.0)
        return (min(corridor, risk_ring), clearance_bonus)

    protected: list[tuple[float, float]] = []
    for point in [start, goal, *skeleton]:
        key = (round(point[0] * 10), round(point[1] * 10))
        if key not in {(round(p[0] * 10), round(p[1] * 10)) for p in protected}:
            protected.append(point)

    unique: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for point in protected + sorted(raw_points, key=priority):
        key = (round(point[0] * 10), round(point[1] * 10))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
        if len(unique) >= args.max_graph_nodes:
            break

    if start not in unique:
        unique.insert(0, start)
    if goal not in unique:
        unique.insert(1, goal)

    coords: dict[str, tuple[float, float]] = {"S": start, "G": goal}
    node_rows = [
        {"id": "S", "x": round(start[0], 2), "y": round(start[1], 2), "clearance": round(_base_clearance(base, *start), 2), "risk": round(_risk_value(start, risk_center, risk_radius), 3)},
        {"id": "G", "x": round(goal[0], 2), "y": round(goal[1], 2), "clearance": round(_base_clearance(base, *goal), 2), "risk": round(_risk_value(goal, risk_center, risk_radius), 3)},
    ]
    idx = 0
    for point in unique:
        if math.hypot(point[0] - start[0], point[1] - start[1]) < 0.15 or math.hypot(point[0] - goal[0], point[1] - goal[1]) < 0.15:
            continue
        node_id = f"v{idx}"
        idx += 1
        coords[node_id] = point
        node_rows.append(
            {
                "id": node_id,
                "x": round(point[0], 2),
                "y": round(point[1], 2),
                "clearance": round(_base_clearance(base, *point), 2),
                "risk": round(_risk_value(point, risk_center, risk_radius), 3),
            }
        )

    ids = list(coords)
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            a = coords[a_id]
            b = coords[b_id]
            dist = math.hypot(a[0] - b[0], a[1] - b[1])
            if dist <= args.edge_radius and _line_free(base, a, b):
                adjacency[a_id].append(b_id)
                adjacency[b_id].append(a_id)
    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda other: math.hypot(coords[other][0] - coords[node_id][0], coords[other][1] - coords[node_id][1]))
        adjacency[node_id] = adjacency[node_id][: args.max_neighbors]
    dropped_edges = 0
    if args.graph_edge_drop_rate > 0:
        drop_rng = random.Random(f"{args.seed}:{case.get('case_id', '')}:edge_drop")
        undirected = sorted({tuple(sorted((a, b))) for a, neighbors in adjacency.items() for b in neighbors if a != b})
        for a, b in undirected:
            if drop_rng.random() < args.graph_edge_drop_rate:
                if b in adjacency.get(a, []):
                    adjacency[a].remove(b)
                if a in adjacency.get(b, []):
                    adjacency[b].remove(a)
                dropped_edges += 1

    return {
        "coords": coords,
        "nodes": node_rows,
        "adjacency": adjacency,
        "start": start,
        "goal": goal,
        "risk_center": risk_center,
        "risk_radius": risk_radius,
        "scenario": scenario,
        "route_constraint": case.get("route_constraint", ""),
        "graph_edge_drop_rate": args.graph_edge_drop_rate,
        "dropped_edges": dropped_edges,
        "risk_center_noise": args.risk_center_noise,
        "risk_radius_scale": args.risk_radius_scale,
    }


def _shortest_repair(graph: dict[str, Any], args) -> list[str]:
    return _shortest_path_from(graph, args, "S")


def _shortest_path_from(graph: dict[str, Any], args, start_node: str) -> list[str]:
    coords: dict[str, tuple[float, float]] = graph["coords"]
    adjacency: dict[str, list[str]] = graph["adjacency"]
    risk_center = graph["risk_center"]
    risk_radius = graph["risk_radius"]
    scenario = graph["scenario"]
    pq: list[tuple[float, str, list[str]]] = [(0.0, start_node, [start_node])]
    best: dict[str, float] = {start_node: 0.0}
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == "G":
            return path
        if cost > best.get(node, float("inf")) + 1e-9:
            continue
        for nxt in adjacency.get(node, []):
            a = coords[node]
            b = coords[nxt]
            edge = math.hypot(a[0] - b[0], a[1] - b[1])
            risk = _risk_value(b, risk_center, risk_radius) if scenario == "semantic_constraint" else 0.0
            step = edge + args.semantic_cost_weight * risk
            new_cost = cost + step
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(pq, (new_cost, nxt, path + [nxt]))
    return [start_node, "G"]


def _graph_shortest_distance(graph: dict[str, Any]) -> list[str]:
    coords: dict[str, tuple[float, float]] = graph["coords"]
    adjacency: dict[str, list[str]] = graph["adjacency"]
    pq: list[tuple[float, str, list[str]]] = [(0.0, "S", ["S"])]
    best: dict[str, float] = {"S": 0.0}
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == "G":
            return path
        if cost > best.get(node, float("inf")) + 1e-9:
            continue
        for nxt in adjacency.get(node, []):
            a = coords[node]
            b = coords[nxt]
            step = math.hypot(a[0] - b[0], a[1] - b[1])
            new_cost = cost + step
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(pq, (new_cost, nxt, path + [nxt]))
    return ["S", "G"]


def _parse_float_list(text: object, default: list[float]) -> list[float]:
    if text in (None, ""):
        return default
    values: list[float] = []
    for chunk in re.split(r"[, ]+", str(text).strip()):
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            continue
    return values or default


def _graph_route_with_risk_weight(graph: dict[str, Any], risk_weight: float, start_node: str = "S") -> list[str]:
    coords: dict[str, tuple[float, float]] = graph["coords"]
    adjacency: dict[str, list[str]] = graph["adjacency"]
    risk_center = graph["risk_center"]
    risk_radius = graph["risk_radius"]
    scenario = graph["scenario"]
    pq: list[tuple[float, str, list[str]]] = [(0.0, start_node, [start_node])]
    best: dict[str, float] = {start_node: 0.0}
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == "G":
            return path
        if cost > best.get(node, float("inf")) + 1e-9:
            continue
        for nxt in adjacency.get(node, []):
            a = coords[node]
            b = coords[nxt]
            edge = math.hypot(a[0] - b[0], a[1] - b[1])
            risk = _risk_value(b, risk_center, risk_radius) if scenario == "semantic_constraint" else 0.0
            new_cost = cost + edge + risk_weight * risk
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(pq, (new_cost, nxt, path + [nxt]))
    return [start_node, "G"]


def _route_metrics_from_nodes(nodes: list[str], graph: dict[str, Any]) -> dict[str, Any]:
    coords = graph["coords"]
    start = graph["start"]
    route = [coords[node] for node in nodes[1:] if node in coords]
    semantic = _semantic_cost(route, graph["risk_center"], graph["risk_radius"]) if graph["scenario"] == "semantic_constraint" else 0.0
    return {
        "route_nodes": nodes,
        "distance": round(_route_length(route, start), 3),
        "turns": _turn_count([start] + route),
        "node_count": max(len(nodes) - 1, 0),
        "semantic_cost": round(semantic, 3),
    }


def _route_option_rows(graph: dict[str, Any], args) -> list[dict[str, Any]]:
    weights = _parse_float_list(args.route_option_risk_weights, [0.0, 1.0, 2.5, 5.0, 10.0])
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for weight in weights:
        nodes = _graph_route_with_risk_weight(graph, float(weight))
        key = tuple(nodes)
        if key in seen:
            continue
        seen.add(key)
        metrics = _route_metrics_from_nodes(nodes, graph)
        tag = str(weight).replace(".", "p").replace("-", "m")
        metrics.update(
            {
                "option_id": f"risk_{tag}",
                "risk_weight": float(weight),
                "valid_graph_route": int(_validate_nodes(nodes, graph, args)[0]),
            }
        )
        rows.append(metrics)
    return rows


def _hop_distances_to_goal(graph: dict[str, Any]) -> dict[str, int]:
    reverse: dict[str, list[str]] = {node: [] for node in graph["coords"]}
    for node, neighbors in graph["adjacency"].items():
        for nxt in neighbors:
            reverse.setdefault(nxt, []).append(node)
    distances = {"G": 0}
    queue = ["G"]
    for node in queue:
        for prev in reverse.get(node, []):
            if prev not in distances:
                distances[prev] = distances[node] + 1
                queue.append(prev)
    return distances


def _candidate_features(
    graph: dict[str, Any],
    current: str,
    node: str,
    current_goal_distance: float,
    hop_distances: dict[str, int],
) -> dict[str, Any]:
    coords: dict[str, tuple[float, float]] = graph["coords"]
    goal_xy = coords["G"]
    current_xy = coords[current]
    node_xy = coords[node]
    distance_to_goal = math.hypot(node_xy[0] - goal_xy[0], node_xy[1] - goal_xy[1])
    edge_distance = math.hypot(node_xy[0] - current_xy[0], node_xy[1] - current_xy[1])
    risk_score = _risk_value(node_xy, graph["risk_center"], graph["risk_radius"]) if graph["scenario"] == "semantic_constraint" else 0.0
    return {
        "id": node,
        "coord": [round(node_xy[0], 2), round(node_xy[1], 2)],
        "is_goal": node == "G",
        "edge_distance": round(edge_distance, 3),
        "distance_to_goal": round(distance_to_goal, 3),
        "progress_delta": round(current_goal_distance - distance_to_goal, 3),
        "hops_to_goal": hop_distances.get(node, 999),
        "risk_score": round(risk_score, 3),
    }


def _heuristic_stepwise_route(graph: dict[str, Any], args, rng: random.Random) -> tuple[list[str], dict[str, Any], str]:
    nodes = ["S"]
    current = "S"
    coords: dict[str, tuple[float, float]] = graph["coords"]
    goal_xy = coords["G"]
    hop_distances = _hop_distances_to_goal(graph)
    max_hops = max(4, min(args.max_llm_subgoals, len(graph["nodes"]) + 2))
    for _hop in range(max_hops):
        allowed = [node for node in graph["adjacency"].get(current, []) if node == "G" or node not in nodes]
        if not allowed:
            break
        if "G" in allowed:
            next_node = "G"
        elif args.planner_mode == "random_legal":
            next_node = rng.choice(sorted(allowed))
        else:
            current_xy = coords[current]
            current_goal_distance = math.hypot(current_xy[0] - goal_xy[0], current_xy[1] - goal_xy[1])
            rows = [_candidate_features(graph, current, node, current_goal_distance, hop_distances) for node in allowed]
            if args.planner_mode == "first_candidate":
                rows.sort(key=lambda item: (not item["is_goal"], item["hops_to_goal"], -item["progress_delta"], item["risk_score"], item["edge_distance"]))
            elif args.planner_mode == "weighted_scorer":
                rows.sort(key=lambda item: _weighted_candidate_score(item, "balanced"))
            elif args.planner_mode == "preference_scorer":
                rows.sort(key=lambda item: _weighted_candidate_score(item, _preference_kind(graph.get("route_constraint", ""))))
            elif args.planner_mode == "greedy_progress":
                rows.sort(key=lambda item: (-item["progress_delta"], item["hops_to_goal"], item["risk_score"], item["edge_distance"]))
            elif args.planner_mode == "greedy_hop":
                rows.sort(key=lambda item: (item["hops_to_goal"], -item["progress_delta"], item["risk_score"], item["edge_distance"]))
            elif args.planner_mode == "greedy_risk":
                rows.sort(key=lambda item: (item["risk_score"], item["hops_to_goal"], -item["progress_delta"], item["edge_distance"]))
            else:
                raise ValueError(f"unknown heuristic planner mode: {args.planner_mode}")
            next_node = str(rows[0]["id"])
        nodes.append(next_node)
        current = next_node
        if current == "G":
            break
    return nodes, {
        "parse_ok": bool(nodes[-1] == "G"),
        "response_source": "heuristic",
        "content_chars": 0,
        "reasoning_chars": 0,
        "step_parse_ok_count": max(len(nodes) - 1, 0),
        "step_count": max(len(nodes) - 1, 0),
        "raw_preview": args.planner_mode,
    }, args.planner_mode


def _preference_kind(text: object) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ("shortest", "quickest", "smallest travel distance", "minimize travel")):
        return "shortest"
    if any(token in lowered for token in ("lowest risk", "risk", "safer", "unsafe", "avoid")):
        return "lowest_risk"
    if any(token in lowered for token in ("clearance", "wide", "narrow", "free space")):
        return "high_clearance"
    if any(token in lowered for token in ("fewest turns", "smooth", "turn", "zig-zag")):
        return "fewest_turns"
    return "balanced"


def _weighted_candidate_score(item: dict[str, Any], preference: str) -> tuple[float, float, float, float]:
    hops = float(item.get("hops_to_goal", 999))
    progress = float(item.get("progress_delta", 0.0))
    risk = float(item.get("risk_score", 0.0))
    edge = float(item.get("edge_distance", 0.0))
    distance = float(item.get("distance_to_goal", 999.0))
    is_goal_bonus = -1000.0 if item.get("is_goal") else 0.0
    if preference == "shortest":
        score = is_goal_bonus + 1.2 * distance + 0.8 * edge + 0.2 * hops + 0.2 * risk - 0.4 * progress
    elif preference == "lowest_risk":
        score = is_goal_bonus + 8.0 * risk + 0.5 * hops + 0.2 * edge - 0.2 * progress
    elif preference == "high_clearance":
        # Clearance is not serialized as a prompt feature in older runs, so this
        # deterministic proxy prefers low risk and short local edges.
        score = is_goal_bonus + 4.0 * risk + 0.5 * edge + 0.5 * hops - 0.3 * progress
    elif preference == "fewest_turns":
        # Without heading history in the compact feature row, hop progress is the
        # strongest deterministic proxy for smooth high-level routes.
        score = is_goal_bonus + 0.9 * hops - 0.7 * progress + 0.2 * edge + 0.2 * risk
    else:
        score = is_goal_bonus + 1.0 * hops - 0.7 * progress + 2.0 * risk + 0.35 * edge
    return (score, hops, -progress, edge)


def _parse_route_nodes(text: str, valid_ids: set[str]) -> tuple[list[str], bool]:
    try:
        parsed = _json_from_text(text)
        candidate = parsed.get("route_nodes") or parsed.get("nodes") or parsed.get("route") or parsed.get("waypoints")
        if isinstance(candidate, list):
            nodes = [str(item) for item in candidate if str(item) in valid_ids]
            return nodes, bool(nodes)
    except Exception:
        pass
    nodes = [tok for tok in re.findall(r"\b(?:S|G|v\d+)\b", text) if tok in valid_ids]
    return nodes, False


def _parse_llm_message(data: dict[str, Any], valid_ids: set[str], parse_reasoning_content: bool) -> tuple[list[str], dict[str, Any], str]:
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = str(message.get("content") or "")
    reasoning_content = str(message.get("reasoning_content") or "")
    candidates = [("content", content)]
    if parse_reasoning_content and reasoning_content:
        candidates.append(("reasoning_content", reasoning_content))

    fallback_nodes: list[str] = []
    fallback_source = "content"
    fallback_text = content
    fallback_parse_ok = False
    for source, text in candidates:
        if not text:
            continue
        nodes, parse_ok = _parse_route_nodes(text, valid_ids)
        if parse_ok:
            return nodes, {
                "parse_ok": True,
                "response_source": source,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning_content),
                "raw_preview": text[:240].replace("\n", " "),
            }, text
        if not fallback_text:
            fallback_nodes = nodes
            fallback_source = source
            fallback_text = text
            fallback_parse_ok = parse_ok

    if not fallback_nodes and fallback_text:
        fallback_nodes, fallback_parse_ok = _parse_route_nodes(fallback_text, valid_ids)
    return fallback_nodes, {
        "parse_ok": fallback_parse_ok,
        "response_source": fallback_source,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning_content),
        "raw_preview": fallback_text[:240].replace("\n", " "),
    }, fallback_text


def _parse_next_node(text: str, allowed: set[str]) -> tuple[str | None, bool]:
    try:
        parsed = _json_from_text(text)
        candidate = parsed.get("next_node") or parsed.get("node") or parsed.get("subgoal")
        if isinstance(candidate, str) and candidate in allowed:
            return candidate, True
        route = parsed.get("route_nodes") or parsed.get("nodes") or parsed.get("route")
        if isinstance(route, list):
            for item in route:
                node = str(item)
                if node in allowed:
                    return node, True
    except Exception:
        pass
    for token in re.findall(r"\b(?:G|v\d+)\b", text):
        if token in allowed:
            return token, False
    return None, False


def _parse_llm_next_node(data: dict[str, Any], allowed: set[str], parse_reasoning_content: bool) -> tuple[str | None, dict[str, Any], str]:
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = str(message.get("content") or "")
    reasoning_content = str(message.get("reasoning_content") or "")
    candidates = [("content", content)]
    if parse_reasoning_content and reasoning_content:
        candidates.append(("reasoning_content", reasoning_content))

    fallback_node: str | None = None
    fallback_source = "content"
    fallback_text = content
    fallback_parse_ok = False
    for source, text in candidates:
        if not text:
            continue
        node, parse_ok = _parse_next_node(text, allowed)
        if node and parse_ok:
            return node, {
                "parse_ok": True,
                "response_source": source,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning_content),
                "raw_preview": text[:240].replace("\n", " "),
            }, text
        if not fallback_text or (node and not fallback_node):
            fallback_node = node
            fallback_source = source
            fallback_text = text
            fallback_parse_ok = parse_ok

    return fallback_node, {
        "parse_ok": fallback_parse_ok,
        "response_source": fallback_source,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning_content),
        "raw_preview": fallback_text[:240].replace("\n", " "),
    }, fallback_text


def _request_map_llm_route(
    graph: dict[str, Any],
    case: dict[str, Any],
    args,
    feedback: str | None = None,
    previous_nodes: list[str] | None = None,
) -> tuple[list[str], dict[str, Any], str]:
    payload = {
        "map_source": "ROS2/Gazebo 3D semantic world projected to a compact route graph",
        "case_id": case.get("case_id"),
        "scenario": graph["scenario"],
        "start_node": "S",
        "goal_node": "G",
        "task": case.get("route_constraint"),
        "risk_zone": {
            "center": [round(graph["risk_center"][0], 2), round(graph["risk_center"][1], 2)],
            "radius": round(graph["risk_radius"], 2),
            "meaning": "semantic/risk/narrow region; avoid when the task asks for safer routing",
        },
        "nodes": graph["nodes"],
        "edges": graph["adjacency"],
        "instruction": (
            "Plan a waypoint route as node ids from S to G. Do not choose a provided route id; no route options are given. "
            "For every consecutive pair [a,b] in route_nodes, b must be listed in edges[a]. "
            "Do not skip intermediate graph nodes just to make the route shorter."
        ),
    }
    if feedback:
        payload["validator_feedback"] = {
            "previous_route_nodes": previous_nodes or [],
            "why_invalid": feedback,
            "repair_instruction": "Return a corrected route_nodes list. Preserve useful nodes only when all adjacent transitions are valid.",
        }
    prompt = {
        "model": args.lm_model,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a high-level waypoint planner for an indoor mobile robot.",
                        "You receive a compact graph converted from a ROS2/Gazebo semantic world.",
                        "Return JSON only with this schema: {\"route_nodes\":[\"<node_id>\",\"...\"],\"reason\":\"short\"}.",
                        "Use only node ids from nodes and only edges listed in edges.",
                        "The route must begin at S and end at G.",
                        "Before answering, verify every consecutive transition: next node must be in edges[current node].",
                        "If the graph requires many intermediate nodes, include them all.",
                        "Avoid high-risk nodes when the task asks for safer routing.",
                    ]
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.llm_max_tokens,
    }
    data = _chat_completion_data(prompt, args)
    return _parse_llm_message(data, set(graph["coords"]), args.parse_reasoning_content)


def _parse_risk_weight_decision(text: str, allowed_weights: list[float]) -> tuple[float, bool]:
    try:
        parsed = _json_from_text(text)
        value = parsed.get("risk_weight")
        if value is None:
            value = parsed.get("semantic_risk_weight")
        if value is not None:
            numeric = float(value)
            return min(allowed_weights, key=lambda item: abs(item - numeric)), True
    except Exception:
        pass
    numbers = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if numbers:
        numeric = numbers[0]
        return min(allowed_weights, key=lambda item: abs(item - numeric)), False
    return allowed_weights[0], False


def _parse_route_option_decision(text: str, allowed_ids: set[str]) -> tuple[str | None, bool]:
    try:
        parsed = _json_from_text(text)
        value = parsed.get("option_id") or parsed.get("route_id") or parsed.get("selected_option")
        if isinstance(value, str) and value in allowed_ids:
            return value, True
    except Exception:
        pass
    for option_id in sorted(allowed_ids, key=len, reverse=True):
        if option_id in text:
            return option_id, False
    return None, False


def _request_language_to_cost_route(graph: dict[str, Any], case: dict[str, Any], args) -> tuple[list[str], dict[str, Any], str]:
    weights = _parse_float_list(args.route_option_risk_weights, [0.0, 1.0, 2.5, 5.0, 10.0])
    payload = {
        "case_id": case.get("case_id"),
        "scenario": graph["scenario"],
        "task": case.get("route_constraint"),
        "risk_weight_options": weights,
        "meaning": (
            "Choose only the semantic risk penalty for a deterministic graph planner. "
            "0 means shortest distance; larger values increasingly avoid semantic-risk nodes."
        ),
    }
    prompt = {
        "model": args.lm_model,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You translate a natural-language navigation preference into one graph-search cost weight.",
                        "Return JSON only with this schema: {\"risk_weight\": <one of the provided options>, \"reason\":\"short\"}.",
                        "Do not output a route. A deterministic Dijkstra planner will compute the route.",
                    ]
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": args.temperature,
        "max_tokens": min(args.llm_max_tokens, 120),
    }
    data = _chat_completion_data(prompt, args)
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = str(message.get("content") or "")
    reasoning_content = str(message.get("reasoning_content") or "")
    parse_text = content or (reasoning_content if args.parse_reasoning_content else "")
    weight, parse_ok = _parse_risk_weight_decision(parse_text, weights)
    nodes = _graph_route_with_risk_weight(graph, weight)
    return nodes, {
        "parse_ok": parse_ok,
        "response_source": f"language_to_cost:risk_weight={weight:g}",
        "content_chars": len(content),
        "reasoning_chars": len(reasoning_content),
        "step_parse_ok_count": int(parse_ok),
        "step_count": 1,
        "raw_preview": parse_text[:240].replace("\n", " "),
        "route_option_id": "",
        "route_risk_weight": weight,
        "llm_route_decision_kind": "language_to_cost",
    }, parse_text


def _request_route_option_rank_route(graph: dict[str, Any], case: dict[str, Any], args) -> tuple[list[str], dict[str, Any], str]:
    options = _route_option_rows(graph, args)
    if not options:
        nodes = _shortest_repair(graph, args)
        return nodes, {
            "parse_ok": False,
            "response_source": "route_option_rank:no_options",
            "content_chars": 0,
            "reasoning_chars": 0,
            "step_parse_ok_count": 0,
            "step_count": 1,
            "raw_preview": "no_options",
            "route_option_id": "",
            "route_risk_weight": "",
            "llm_route_decision_kind": "route_option_rank",
        }, ""
    prompt_options = [
        {
            "option_id": row["option_id"],
            "risk_weight": row["risk_weight"],
            "route_nodes": row["route_nodes"],
            "distance": row["distance"],
            "turns": row["turns"],
            "semantic_cost": row["semantic_cost"],
            "node_count": row["node_count"],
        }
        for row in options
    ]
    payload = {
        "case_id": case.get("case_id"),
        "scenario": graph["scenario"],
        "task": case.get("route_constraint"),
        "route_options": prompt_options,
        "instruction": "Choose one option_id. Do not invent a route or modify route_nodes.",
    }
    prompt = {
        "model": args.lm_model,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are selecting among deterministic graph-planner route options for a mobile robot.",
                        "Return JSON only with this schema: {\"option_id\":\"<provided option_id>\",\"reason\":\"short\"}.",
                        "Use the natural-language task, distance, turns, and semantic_cost to choose.",
                    ]
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": args.temperature,
        "max_tokens": min(args.llm_max_tokens, 160),
    }
    data = _chat_completion_data(prompt, args)
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = str(message.get("content") or "")
    reasoning_content = str(message.get("reasoning_content") or "")
    parse_text = content or (reasoning_content if args.parse_reasoning_content else "")
    option_id, parse_ok = _parse_route_option_decision(parse_text, {str(row["option_id"]) for row in options})
    selected = next((row for row in options if row["option_id"] == option_id), options[0])
    nodes = [str(node) for node in selected["route_nodes"]]
    return nodes, {
        "parse_ok": parse_ok,
        "response_source": f"route_option_rank:{selected['option_id']}",
        "content_chars": len(content),
        "reasoning_chars": len(reasoning_content),
        "step_parse_ok_count": int(parse_ok),
        "step_count": 1,
        "raw_preview": parse_text[:240].replace("\n", " "),
        "route_option_id": selected["option_id"],
        "route_risk_weight": selected["risk_weight"],
        "llm_route_decision_kind": "route_option_rank",
    }, parse_text


def _request_stepwise_llm_route(graph: dict[str, Any], case: dict[str, Any], args) -> tuple[list[str], dict[str, Any], str]:
    nodes = ["S"]
    current = "S"
    raw_parts: list[str] = []
    parse_ok_steps = 0
    content_chars = 0
    reasoning_chars = 0
    response_source = "content"
    raw_preview = ""
    order_gate_steps = 0
    order_gate_accepts = 0
    order_gate_fallbacks = 0
    order_gate_consistency_sum = 0.0
    order_gate_vote_entropy_sum = 0.0
    max_hops = max(4, min(args.max_llm_subgoals, len(graph["nodes"]) + 2))
    coords = graph["coords"]
    goal_xy = coords["G"]
    hop_distances = _hop_distances_to_goal(graph)
    for hop in range(max_hops):
        allowed = [node for node in graph["adjacency"].get(current, []) if node == "G" or node not in nodes]
        if not allowed:
            break
        current_xy = coords[current]
        current_goal_distance = math.hypot(current_xy[0] - goal_xy[0], current_xy[1] - goal_xy[1])
        canonical_rows = [_candidate_features(graph, current, node, current_goal_distance, hop_distances) for node in allowed]
        canonical_rows.sort(key=lambda item: (not item["is_goal"], item["hops_to_goal"], -item["progress_delta"], item["risk_score"]))

        def ablate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = [dict(row) for row in rows]
            if args.candidate_feature_ablation == "no_risk":
                for row in out:
                    row.pop("risk_score", None)
            elif args.candidate_feature_ablation == "no_hop":
                for row in out:
                    row.pop("hops_to_goal", None)
            elif args.candidate_feature_ablation == "no_progress":
                for row in out:
                    row.pop("progress_delta", None)
            elif args.candidate_feature_ablation == "shuffle_order":
                random.Random(args.seed + hop).shuffle(out)
            return out

        candidate_rows = ablate_rows(canonical_rows)

        def make_prompt(rows: list[dict[str, Any]]) -> dict[str, Any]:
            payload = {
                "map_source": "ROS2/Gazebo 3D semantic world projected to a compact route graph",
                "case_id": case.get("case_id"),
                "scenario": graph["scenario"],
                "task": case.get("route_constraint"),
                "current_node": current,
                "current_distance_to_goal": round(current_goal_distance, 3),
                "step_index": hop,
                "goal_node": "G",
                "visited_nodes": nodes,
                "allowed_next_nodes": rows,
                "risk_zone": {
                    "center": [round(graph["risk_center"][0], 2), round(graph["risk_center"][1], 2)],
                    "radius": round(graph["risk_radius"], 2),
                    "meaning": "semantic/risk/narrow region; avoid when the task asks for safer routing",
                },
                "instruction": (
                    "Choose exactly one next subgoal node from allowed_next_nodes. "
                    "Candidate order is a presentation detail, not a route ranking; compare the numeric fields. "
                    "This is receding-horizon path planning for a small edge LLM; do not invent nodes and do not output a full route. "
                    "If G appears in allowed_next_nodes, choose G. Otherwise prefer candidates with positive progress_delta, fewer hops_to_goal, "
                    "low risk_score, and short edge_distance."
                ),
            }
            return {
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
                                "Otherwise choose a node that reduces distance_to_goal and hops_to_goal while avoiding risk and loops.",
                                "Do not choose a node with negative progress_delta unless every allowed candidate has negative progress_delta.",
                                "Do not use the candidate list order as the main criterion; use the candidate features.",
                            ]
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "temperature": args.temperature,
                "max_tokens": args.llm_step_max_tokens,
            }

        def query_prompt(prompt: dict[str, Any], attempts: int) -> tuple[str | None, dict[str, Any], str]:
            chosen: str | None = None
            meta: dict[str, Any] = {}
            raw_text = ""
            for attempt in range(attempts):
                if attempt > 0:
                    prompt["messages"].append(
                        {
                            "role": "user",
                            "content": "The previous answer was invalid or did not choose exactly one allowed node. Return JSON only and choose one id from allowed_next_nodes.",
                        }
                    )
                data = _chat_completion_data(prompt, args)
                chosen, meta, raw_text = _parse_llm_next_node(data, set(allowed), args.parse_reasoning_content)
                if chosen:
                    break
            return chosen, meta, raw_text

        def order_row_variants() -> list[list[dict[str, Any]]]:
            row_variants: list[list[dict[str, Any]]] = [candidate_rows]
            if args.planner_mode == "llm_step_consistency_gate":
                for offset in range(1, max(1, int(args.order_gate_variants))):
                    shuffled = [dict(row) for row in canonical_rows]
                    random.Random(args.seed + hop + 997 * offset).shuffle(shuffled)
                    row_variants.append(ablate_rows(shuffled))
            elif args.candidate_feature_ablation == "shuffle_order":
                for offset in (101, 202):
                    shuffled = [dict(row) for row in canonical_rows]
                    random.Random(args.seed + hop + offset).shuffle(shuffled)
                    row_variants.append(shuffled)
            else:
                risk_sorted = [dict(row) for row in canonical_rows]
                risk_sorted.sort(key=lambda item: (not item["is_goal"], item.get("risk_score", 0.0), item.get("hops_to_goal", 999), -item.get("progress_delta", 0.0)))
                progress_sorted = [dict(row) for row in canonical_rows]
                progress_sorted.sort(key=lambda item: (not item["is_goal"], -item.get("progress_delta", 0.0), item.get("hops_to_goal", 999), item.get("risk_score", 0.0)))
                row_variants.extend([risk_sorted, progress_sorted])
            return row_variants[: max(1, int(args.order_gate_variants))]

        if args.planner_mode in ("llm_step_order_ensemble", "llm_step_consistency_gate"):
            row_variants = order_row_variants()
            votes: list[str] = []
            meta_items: list[dict[str, Any]] = []
            raw_items: list[str] = []
            for rows in row_variants:
                chosen, meta, raw = query_prompt(make_prompt(rows), attempts=1)
                if chosen:
                    votes.append(chosen)
                meta_items.append(meta)
                raw_items.append(raw)
            counts = {node: votes.count(node) for node in set(votes)}
            total_queries = max(len(row_variants), 1)
            top_count = max(counts.values()) if counts else 0
            consistency = top_count / total_queries
            vote_entropy = 0.0
            for count in counts.values():
                p = count / max(len(votes), 1)
                if p > 0:
                    vote_entropy -= p * math.log(p, 2)
            if votes:
                canonical_rank = {str(row["id"]): idx for idx, row in enumerate(canonical_rows)}
                next_node = sorted(votes, key=lambda node: (-counts[node], canonical_rank.get(node, 999)))[0]
            else:
                next_node = None
            if args.planner_mode == "llm_step_consistency_gate":
                order_gate_steps += 1
                order_gate_consistency_sum += consistency
                order_gate_vote_entropy_sum += vote_entropy
                if (not next_node) or top_count < args.order_gate_min_votes or consistency < args.order_gate_min_consistency:
                    order_gate_fallbacks += 1
                    fallback_path = _shortest_path_from(graph, args, current)
                    if len(fallback_path) > 1:
                        nodes.extend(fallback_path[1:])
                        current = nodes[-1]
                    response_source = "order_consistency_gate_fallback"
                    raw_preview = f"votes={votes}; fallback_path={fallback_path}"
                    raw_parts.append("\n--- gated fallback ---\n".join(raw_items + [raw_preview]))
                    break
                order_gate_accepts += 1
            meta = {
                "parse_ok": bool(votes),
                "response_source": "order_consistency_gate" if args.planner_mode == "llm_step_consistency_gate" else "order_ensemble",
                "content_chars": sum(int(item.get("content_chars", 0) or 0) for item in meta_items),
                "reasoning_chars": sum(int(item.get("reasoning_chars", 0) or 0) for item in meta_items),
                "raw_preview": " | ".join(str(item.get("raw_preview", ""))[:80] for item in meta_items),
                "ensemble_votes": ",".join(votes),
                "order_gate_consistency": consistency,
                "order_gate_vote_entropy": vote_entropy,
                "order_gate_fallback": 0,
            }
            raw = "\n--- ensemble ---\n".join(raw_items)
        else:
            prompt = make_prompt(candidate_rows)
            attempts = 1 + max(0, args.llm_retries if args.planner_mode == "llm_step_retry" else 0)
            next_node, meta, raw = query_prompt(prompt, attempts=attempts)
        raw_parts.append(raw)
        raw_preview = str(meta.get("raw_preview", raw_preview))
        response_source = str(meta.get("response_source", response_source))
        content_chars += int(meta.get("content_chars", 0) or 0)
        reasoning_chars += int(meta.get("reasoning_chars", 0) or 0)
        parse_ok_steps += int(bool(meta.get("parse_ok", False)))
        if not next_node:
            break
        nodes.append(next_node)
        current = next_node
        if current == "G":
            break
    return nodes, {
        "parse_ok": bool(parse_ok_steps and nodes[-1] == "G"),
        "response_source": response_source,
        "content_chars": content_chars,
        "reasoning_chars": reasoning_chars,
        "step_parse_ok_count": parse_ok_steps,
        "step_count": max(len(nodes) - 1, 0),
        "raw_preview": raw_preview,
        "order_gate_steps": order_gate_steps,
        "order_gate_accepts": order_gate_accepts,
        "order_gate_fallbacks": order_gate_fallbacks,
        "order_gate_mean_consistency": order_gate_consistency_sum / max(order_gate_steps, 1),
        "order_gate_mean_vote_entropy": order_gate_vote_entropy_sum / max(order_gate_steps, 1),
    }, "\n".join(raw_parts)


def _validate_nodes(nodes: list[str], graph: dict[str, Any], args) -> tuple[bool, str]:
    adjacency = graph["adjacency"]
    if not nodes:
        return False, "empty"
    if nodes[0] != "S":
        return False, "missing_start"
    if nodes[-1] != "G":
        return False, "missing_goal"
    for a, b in zip(nodes[:-1], nodes[1:]):
        if b not in adjacency.get(a, []):
            return False, f"missing_edge:{a}->{b}"
    if graph["scenario"] == "semantic_constraint":
        coords = graph["coords"]
        route = [coords[n] for n in nodes[1:] if n in coords]
        semantic = _semantic_cost(route, graph["risk_center"], graph["risk_radius"])
        if semantic > args.max_planned_semantic_cost:
            return False, f"semantic_cost:{semantic:.3f}"
    return True, "ok"


def _failure_category(reason: str) -> str:
    if reason == "ok":
        return "ok"
    if reason.startswith("llm_error"):
        return "llm_error"
    if reason.startswith("missing_edge"):
        return "missing_edge"
    if reason.startswith("semantic_cost"):
        return "semantic_violation"
    if reason in {"empty", "missing_start", "missing_goal"}:
        return reason
    if reason == "no_llm_graph_search":
        return "no_llm_graph_search"
    return "other_invalid"


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        cur = [i]
        for j, right in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + int(left != right),
                )
            )
        prev = cur
    return prev[-1]


def _retained_node_fraction(raw_nodes: list[str], exec_nodes: list[str]) -> float:
    if not raw_nodes:
        return 0.0
    exec_set = set(exec_nodes)
    return sum(1 for node in raw_nodes if node in exec_set) / len(raw_nodes)


def _nodes_to_route(nodes: list[str], graph: dict[str, Any]) -> list[tuple[float, float]]:
    coords = graph["coords"]
    return [coords[node] for node in nodes[1:] if node in coords]


def _nodes_to_raw_route(nodes: list[str], graph: dict[str, Any]) -> list[tuple[float, float]]:
    coords = graph["coords"]
    return [coords[node] for node in nodes if node != "S" and node in coords]


def _sanitize_raw_route(env, raw_waypoints: list[tuple[float, float]], max_segment: float = 5.8) -> tuple[list[tuple[float, float]], bool]:
    route: list[tuple[float, float]] = []
    cursor = (float(env.robot_x), float(env.robot_y))
    valid = True
    for point in raw_waypoints:
        if env._in_collision(point[0], point[1]):
            valid = False
            continue
        if math.hypot(point[0] - cursor[0], point[1] - cursor[1]) > max_segment:
            valid = False
        route.append(point)
        cursor = point
    return route[:28], valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ppo", "sac"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
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
    parser.add_argument("--lm-studio-url", default=os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234"))
    parser.add_argument("--lm-model", default=os.environ.get("LM_STUDIO_MODEL", "lfm2.5-8b-a1b-mlx"))
    parser.add_argument("--llm-timeout-s", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=350)
    parser.add_argument("--llm-step-max-tokens", type=int, default=96)
    parser.add_argument("--max-llm-subgoals", type=int, default=28)
    parser.add_argument("--no-parse-reasoning-content", dest="parse_reasoning_content", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--graph-resolution", type=float, default=2.2)
    parser.add_argument("--waypoint-spacing", type=float, default=2.2)
    parser.add_argument("--waypoint-grid-resolution", type=float, default=0.55)
    parser.add_argument("--graph-margin", type=float, default=4.5)
    parser.add_argument("--edge-radius", type=float, default=3.5)
    parser.add_argument("--max-graph-nodes", type=int, default=42)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--semantic-cost-weight", type=float, default=5.0)
    parser.add_argument("--max-planned-semantic-cost", type=float, default=0.75)
    parser.add_argument("--graph-edge-drop-rate", type=float, default=0.0)
    parser.add_argument("--risk-center-noise", type=float, default=0.0)
    parser.add_argument("--risk-radius-scale", type=float, default=1.0)
    parser.add_argument("--repair-invalid-route", action="store_true")
    parser.add_argument(
        "--planner-mode",
        choices=(
            "llm",
            "llm_retry",
            "llm_raw",
            "llm_step",
            "llm_step_retry",
            "llm_step_order_ensemble",
            "llm_step_consistency_gate",
            "no_llm",
            "graph_shortest",
            "first_candidate",
            "weighted_scorer",
            "preference_scorer",
            "language_to_cost",
            "route_option_rank",
            "greedy_progress",
            "greedy_hop",
            "greedy_risk",
            "random_legal",
        ),
        default="llm",
    )
    parser.add_argument("--llm-retries", type=int, default=0)
    parser.add_argument("--order-gate-variants", type=int, default=3)
    parser.add_argument("--order-gate-min-votes", type=int, default=2)
    parser.add_argument("--order-gate-min-consistency", type=float, default=0.67)
    parser.add_argument("--route-option-risk-weights", default="0,1,2.5,5,10")
    parser.add_argument("--candidate-feature-ablation", choices=("none", "no_risk", "no_hop", "no_progress", "shuffle_order"), default="none")
    parser.add_argument("--episode-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()
    if args.planner_mode == "llm_retry" and args.llm_retries <= 0:
        args.llm_retries = 2

    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for evaluation.") from exc

    cases = _load_cases(args.cases)[: args.episodes]
    model_cls = PPO if args.algo == "ppo" else SAC
    model = model_cls.load(args.model)

    print(f"Map-conditioned planning mode={args.planner_mode} model={args.lm_model} algo={args.algo}")
    print(f"cases={args.cases} episodes={len(cases)}")
    successes = collisions = timeouts = invalid_routes = repaired_routes = 0
    strict_valid = parse_ok_count = 0
    no_llm_routes = retry_successes = raw_execution_count = 0
    total_reward = 0.0
    steps_all: list[int] = []
    steps_success: list[int] = []
    semantic_costs: list[float] = []
    route_lengths: list[float] = []
    route_turns: list[int] = []
    route_scores: list[float] = []
    edit_distances: list[int] = []
    retained_fracs: list[float] = []
    order_gate_steps_total = 0
    order_gate_accepts_total = 0
    order_gate_fallbacks_total = 0
    order_gate_consistency_values: list[float] = []
    order_gate_entropy_values: list[float] = []
    failure_counts: dict[str, int] = {}
    episode_rows: list[dict[str, Any]] = []

    for episode, case in enumerate(cases):
        env = _make_env(args, str(case.get("map_id") or args.maps[0]), args.seed + episode)
        obs, start, final_goal, _map_id = _apply_case_reset(env, case, args.seed + episode)
        base = _base_env(env)
        graph = _build_compact_graph(base, case, args)
        raw_nodes: list[str] = []
        nodes: list[str] = []
        valid = False
        repair = 0
        parse_ok = 0
        invalid_reason = "ok"
        raw_preview = ""
        response_source = "content"
        content_chars = 0
        reasoning_chars = 0
        step_parse_ok_count = 0
        step_count = 0
        retry_count = 0
        raw_execution = 0
        order_gate_steps = 0
        order_gate_accepts = 0
        order_gate_fallbacks = 0
        order_gate_mean_consistency = 0.0
        order_gate_mean_vote_entropy = 0.0
        route_option_id = ""
        route_risk_weight: float | str = ""
        llm_route_decision_kind = ""
        if args.planner_mode == "no_llm":
            invalid_reason = "no_llm_graph_search"
            nodes = _shortest_repair(graph, args)
            valid, invalid_reason = _validate_nodes(nodes, graph, args)
            no_llm_routes += 1
            if valid:
                strict_valid += 1
        elif args.planner_mode == "graph_shortest":
            invalid_reason = "graph_shortest_distance"
            nodes = _graph_shortest_distance(graph)
            valid, invalid_reason = _validate_nodes(nodes, graph, args)
            no_llm_routes += 1
            if valid:
                strict_valid += 1
        elif args.planner_mode in ("first_candidate", "weighted_scorer", "preference_scorer", "greedy_progress", "greedy_hop", "greedy_risk", "random_legal"):
            raw_nodes, meta, _raw = _heuristic_stepwise_route(graph, args, random.Random(args.seed + episode))
            raw_preview = str(meta.get("raw_preview", ""))
            response_source = str(meta.get("response_source", "heuristic"))
            step_parse_ok_count = int(meta.get("step_parse_ok_count", 0) or 0)
            step_count = int(meta.get("step_count", 0) or 0)
            parse_ok = int(bool(meta.get("parse_ok", False)))
            valid, invalid_reason = _validate_nodes(raw_nodes, graph, args)
            nodes = raw_nodes
            no_llm_routes += 1
            if valid:
                strict_valid += 1
        elif args.planner_mode == "language_to_cost":
            try:
                raw_nodes, meta, _raw = _request_language_to_cost_route(graph, case, args)
                raw_preview = str(meta.get("raw_preview", ""))
                response_source = str(meta.get("response_source", "content"))
                content_chars = int(meta.get("content_chars", 0) or 0)
                reasoning_chars = int(meta.get("reasoning_chars", 0) or 0)
                step_parse_ok_count = int(meta.get("step_parse_ok_count", 0) or 0)
                step_count = int(meta.get("step_count", 0) or 0)
                parse_ok = int(bool(meta.get("parse_ok", False)))
                route_option_id = str(meta.get("route_option_id", ""))
                route_risk_weight = meta.get("route_risk_weight", "")
                llm_route_decision_kind = str(meta.get("llm_route_decision_kind", "language_to_cost"))
                valid, invalid_reason = _validate_nodes(raw_nodes, graph, args)
                nodes = raw_nodes
                if valid:
                    strict_valid += 1
            except Exception as exc:
                invalid_reason = f"llm_error:{exc}"
        elif args.planner_mode == "route_option_rank":
            try:
                raw_nodes, meta, _raw = _request_route_option_rank_route(graph, case, args)
                raw_preview = str(meta.get("raw_preview", ""))
                response_source = str(meta.get("response_source", "content"))
                content_chars = int(meta.get("content_chars", 0) or 0)
                reasoning_chars = int(meta.get("reasoning_chars", 0) or 0)
                step_parse_ok_count = int(meta.get("step_parse_ok_count", 0) or 0)
                step_count = int(meta.get("step_count", 0) or 0)
                parse_ok = int(bool(meta.get("parse_ok", False)))
                route_option_id = str(meta.get("route_option_id", ""))
                route_risk_weight = meta.get("route_risk_weight", "")
                llm_route_decision_kind = str(meta.get("llm_route_decision_kind", "route_option_rank"))
                valid, invalid_reason = _validate_nodes(raw_nodes, graph, args)
                nodes = raw_nodes
                if valid:
                    strict_valid += 1
            except Exception as exc:
                invalid_reason = f"llm_error:{exc}"
        elif args.planner_mode in LLM_STEP_MODES:
            try:
                raw_nodes, meta, _raw = _request_stepwise_llm_route(graph, case, args)
                raw_preview = str(meta.get("raw_preview", ""))
                response_source = str(meta.get("response_source", "content"))
                content_chars = int(meta.get("content_chars", 0) or 0)
                reasoning_chars = int(meta.get("reasoning_chars", 0) or 0)
                step_parse_ok_count = int(meta.get("step_parse_ok_count", 0) or 0)
                step_count = int(meta.get("step_count", 0) or 0)
                order_gate_steps = int(meta.get("order_gate_steps", 0) or 0)
                order_gate_accepts = int(meta.get("order_gate_accepts", 0) or 0)
                order_gate_fallbacks = int(meta.get("order_gate_fallbacks", 0) or 0)
                order_gate_mean_consistency = float(meta.get("order_gate_mean_consistency", 0.0) or 0.0)
                order_gate_mean_vote_entropy = float(meta.get("order_gate_mean_vote_entropy", 0.0) or 0.0)
                parse_ok = int(bool(meta.get("parse_ok", False)))
                valid, invalid_reason = _validate_nodes(raw_nodes, graph, args)
                nodes = raw_nodes
                if valid:
                    strict_valid += 1
            except Exception as exc:
                invalid_reason = f"llm_error:{exc}"
        else:
            try:
                attempts = 1 + max(0, args.llm_retries if args.planner_mode in ("llm_retry", "llm_raw") else 0)
                for attempt in range(attempts):
                    feedback = invalid_reason if attempt > 0 else None
                    previous_nodes = raw_nodes if attempt > 0 else None
                    raw_nodes, meta, _raw = _request_map_llm_route(graph, case, args, feedback, previous_nodes)
                    raw_preview = str(meta.get("raw_preview", ""))
                    response_source = str(meta.get("response_source", "content"))
                    content_chars = int(meta.get("content_chars", 0) or 0)
                    reasoning_chars = int(meta.get("reasoning_chars", 0) or 0)
                    parse_ok = int(bool(meta.get("parse_ok", False)))
                    valid, invalid_reason = _validate_nodes(raw_nodes, graph, args)
                    nodes = raw_nodes
                    retry_count = attempt
                    if valid:
                        if attempt > 0:
                            retry_successes += 1
                        break
                if valid:
                    strict_valid += 1
                elif args.planner_mode == "llm_raw":
                    raw_execution = 1
                elif args.repair_invalid_route:
                    nodes = _shortest_repair(graph, args)
                    repair = 1
                    repaired_routes += 1
            except Exception as exc:
                invalid_reason = f"llm_error:{exc}"
                if args.repair_invalid_route and args.planner_mode != "llm_raw":
                    nodes = _shortest_repair(graph, args)
                    repair = 1
                    repaired_routes += 1

        parse_ok_count += parse_ok
        order_gate_steps_total += order_gate_steps
        order_gate_accepts_total += order_gate_accepts
        order_gate_fallbacks_total += order_gate_fallbacks
        if order_gate_steps:
            order_gate_consistency_values.append(order_gate_mean_consistency)
            order_gate_entropy_values.append(order_gate_mean_vote_entropy)
        failure_category = _failure_category(invalid_reason if not valid else "ok")
        if failure_category != "ok":
            failure_counts[failure_category] = failure_counts.get(failure_category, 0) + 1
        edit_distance = _levenshtein(raw_nodes, nodes) if raw_nodes or nodes else 0
        retained_fraction = _retained_node_fraction(raw_nodes, nodes)
        edit_distances.append(edit_distance)
        retained_fracs.append(retained_fraction)
        if invalid_reason.startswith("llm_error") and args.planner_mode in {"language_to_cost", "route_option_rank"}:
            invalid_routes += 1
            print(
                f"episode={episode:03d} case_id={case.get('case_id')} outcome=invalid_route "
                f"plan_valid=0 repaired=0 parse_ok={parse_ok} source={response_source} "
                f"content_chars={content_chars} reasoning_chars={reasoning_chars} raw_nodes={raw_nodes} reason={invalid_reason}"
            )
            episode_rows.append(
                {
                    "run_label": args.run_label,
                    "planner_mode": args.planner_mode,
                    "algo": args.algo,
                    "case_id": case.get("case_id"),
                    "episode": episode,
                    "map_id": case.get("map_id") or _map_id,
                    "seed": args.seed,
                    "scenario": graph["scenario"],
                    "outcome": "invalid_route",
                    "success": 0,
                    "collision": 0,
                    "timeout": 0,
                    "steps": 0,
                    "plan_valid": 0,
                    "parse_ok": parse_ok,
                    "response_source": response_source,
                    "content_chars": content_chars,
                    "reasoning_chars": reasoning_chars,
                    "step_parse_ok_count": step_parse_ok_count,
                    "step_count": step_count,
                    "repaired": 0,
                    "raw_execution": raw_execution,
                    "retry_count": retry_count,
                    "failure_reason": invalid_reason,
                    "failure_category": failure_category,
                    "raw_node_count": len(raw_nodes),
                    "exec_node_count": len(nodes),
                    "node_retention": retained_fraction,
                    "edit_distance": edit_distance,
                    "route_distance": "",
                    "route_turns": "",
                    "semantic_cost": "",
                    "route_score": "",
                    "graph_nodes": len(graph["nodes"]),
                    "dropped_edges": graph.get("dropped_edges", 0),
                    "graph_edge_drop_rate": graph.get("graph_edge_drop_rate", 0.0),
                    "risk_center_noise": graph.get("risk_center_noise", 0.0),
                    "risk_radius_scale": graph.get("risk_radius_scale", 1.0),
                    "order_gate_steps": order_gate_steps,
                    "order_gate_accepts": order_gate_accepts,
                    "order_gate_fallbacks": order_gate_fallbacks,
                    "order_gate_mean_consistency": order_gate_mean_consistency,
                    "order_gate_mean_vote_entropy": order_gate_mean_vote_entropy,
                    "route_option_id": route_option_id,
                    "route_risk_weight": route_risk_weight,
                    "llm_route_decision_kind": llm_route_decision_kind,
                }
            )
            continue
        if not valid and args.planner_mode in ({"llm_raw"} | LLM_STEP_MODES):
            nodes = raw_nodes
            raw_execution = int(args.planner_mode == "llm_raw")
        raw_execution_count += raw_execution
        if not valid and not args.repair_invalid_route and args.planner_mode not in ({"llm_raw"} | LLM_STEP_MODES):
            invalid_routes += 1
            print(
                f"episode={episode:03d} case_id={case.get('case_id')} outcome=invalid_route "
                f"plan_valid=0 repaired=0 parse_ok={parse_ok} source={response_source} "
                f"content_chars={content_chars} reasoning_chars={reasoning_chars} raw_nodes={raw_nodes} reason={invalid_reason}"
            )
            episode_rows.append(
                {
                    "run_label": args.run_label,
                    "planner_mode": args.planner_mode,
                    "algo": args.algo,
                    "case_id": case.get("case_id"),
                    "episode": episode,
                    "map_id": case.get("map_id") or _map_id,
                    "seed": args.seed,
                    "scenario": graph["scenario"],
                    "outcome": "invalid_route",
                    "success": 0,
                    "collision": 0,
                    "timeout": 0,
                    "steps": 0,
                    "plan_valid": 0,
                    "parse_ok": parse_ok,
                    "response_source": response_source,
                    "content_chars": content_chars,
                    "reasoning_chars": reasoning_chars,
                    "step_parse_ok_count": step_parse_ok_count,
                    "step_count": step_count,
                    "repaired": 0,
                    "raw_execution": raw_execution,
                    "retry_count": retry_count,
                    "failure_reason": invalid_reason,
                    "failure_category": failure_category,
                    "raw_node_count": len(raw_nodes),
                    "exec_node_count": len(nodes),
                    "node_retention": retained_fraction,
                    "edit_distance": edit_distance,
                    "route_distance": "",
                    "route_turns": "",
                    "semantic_cost": "",
                    "route_score": "",
                    "graph_nodes": len(graph["nodes"]),
                    "dropped_edges": graph.get("dropped_edges", 0),
                    "graph_edge_drop_rate": graph.get("graph_edge_drop_rate", 0.0),
                    "risk_center_noise": graph.get("risk_center_noise", 0.0),
                    "risk_radius_scale": graph.get("risk_radius_scale", 1.0),
                    "order_gate_steps": order_gate_steps,
                    "order_gate_accepts": order_gate_accepts,
                    "order_gate_fallbacks": order_gate_fallbacks,
                    "order_gate_mean_consistency": order_gate_mean_consistency,
                    "order_gate_mean_vote_entropy": order_gate_mean_vote_entropy,
                    "route_option_id": route_option_id,
                    "route_risk_weight": route_risk_weight,
                    "llm_route_decision_kind": llm_route_decision_kind,
                }
            )
            continue

        route = _nodes_to_raw_route(nodes, graph) if args.planner_mode in ({"llm_raw"} | LLM_STEP_MODES) else _nodes_to_route(nodes, graph)
        if args.planner_mode in ({"llm_raw"} | LLM_STEP_MODES):
            route, geometry_valid = _sanitize_raw_route(base, route)
        else:
            route, geometry_valid = _sanitize_route(base, route, final_goal)
        if not geometry_valid and not args.repair_invalid_route and args.planner_mode not in ({"llm_raw"} | LLM_STEP_MODES):
            invalid_routes += 1
            continue
        if not route and args.planner_mode in ({"llm_raw"} | LLM_STEP_MODES):
            invalid_routes += 1
            continue
        if not route:
            route = [final_goal]

        trajectory: list[tuple[float, float]] = [(float(base.robot_x), float(base.robot_y))]
        episode_reward = 0.0
        steps_used = 0
        collided = False
        for subgoal in route:
            _set_goal(env, subgoal)
            obs = _current_obs(env)
            while steps_used < args.max_steps:
                action = _predict_action(model, obs)
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                episode_reward += float(reward)
                steps_used += 1
                base = _base_env(env)
                trajectory.append((float(base.robot_x), float(base.robot_y)))
                collided = bool(info.get("collided", False))
                if collided:
                    break
                if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) <= base.success_radius:
                    break
                if done:
                    break
            if collided or steps_used >= args.max_steps:
                break
            if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) > base.success_radius:
                break

        base = _base_env(env)
        final_dist = math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1])
        if final_dist <= base.success_radius and not collided:
            successes += 1
            outcome = "success"
            steps_success.append(steps_used)
        elif collided:
            collisions += 1
            outcome = "collision"
        else:
            timeouts += 1
            outcome = "timeout"
        total_reward += episode_reward
        steps_all.append(steps_used)
        risk_center = graph["risk_center"]
        risk_radius = graph["risk_radius"]
        sem = _trajectory_semantic_cost(trajectory, risk_center, risk_radius) if graph["scenario"] == "semantic_constraint" else 0.0
        semantic_costs.append(sem)
        route_len = _route_length(route, start)
        turns = _turn_count([start] + route)
        route_score = _graph_score(nodes, graph["coords"], start, risk_center, risk_radius, graph["scenario"])
        route_lengths.append(route_len)
        route_turns.append(turns)
        route_scores.append(route_score)
        print(
            f"episode={episode:03d} case_id={case.get('case_id')} scenario={graph['scenario']} outcome={outcome:9s} "
            f"reward={episode_reward:8.2f} final_dist={final_dist:6.2f} steps={steps_used:04d} "
            f"route_len={len(route):02d} route_distance={route_len:6.2f} route_turns={turns:02d} "
            f"semantic_cost={sem:.3f} plan_valid={int(valid)} repaired={repair} parse_ok={parse_ok} "
            f"raw_execution={raw_execution} source={response_source} content_chars={content_chars} reasoning_chars={reasoning_chars} "
            f"step_parse_ok={step_parse_ok_count}/{step_count} "
            f"retry_count={retry_count} edit_distance={edit_distance} retained={retained_fraction:.3f} "
            f"geometry_valid={int(bool(geometry_valid))} graph_nodes={len(graph['nodes']):02d} raw_nodes={raw_nodes} exec_nodes={nodes} "
            f"reason={invalid_reason} raw_preview={raw_preview[:120]!r}"
        )
        episode_rows.append(
            {
                "run_label": args.run_label,
                "planner_mode": args.planner_mode,
                "algo": args.algo,
                "case_id": case.get("case_id"),
                "episode": episode,
                "map_id": case.get("map_id") or _map_id,
                "seed": args.seed,
                "scenario": graph["scenario"],
                "outcome": outcome,
                "success": int(outcome == "success"),
                "collision": int(outcome == "collision"),
                "timeout": int(outcome == "timeout"),
                "steps": steps_used,
                "plan_valid": int(valid),
                "parse_ok": parse_ok,
                "response_source": response_source,
                "content_chars": content_chars,
                "reasoning_chars": reasoning_chars,
                "step_parse_ok_count": step_parse_ok_count,
                "step_count": step_count,
                "order_gate_steps": order_gate_steps,
                "order_gate_accepts": order_gate_accepts,
                "order_gate_fallbacks": order_gate_fallbacks,
                "order_gate_mean_consistency": order_gate_mean_consistency,
                "order_gate_mean_vote_entropy": order_gate_mean_vote_entropy,
                "route_option_id": route_option_id,
                "route_risk_weight": route_risk_weight,
                "llm_route_decision_kind": llm_route_decision_kind,
                "repaired": repair,
                "raw_execution": raw_execution,
                "retry_count": retry_count,
                "failure_reason": invalid_reason,
                "failure_category": failure_category,
                "raw_node_count": len(raw_nodes),
                "exec_node_count": len(nodes),
                "node_retention": retained_fraction,
                "edit_distance": edit_distance,
                "route_distance": route_len,
                "route_turns": turns,
                "semantic_cost": sem,
                "route_score": route_score,
                "graph_nodes": len(graph["nodes"]),
                "dropped_edges": graph.get("dropped_edges", 0),
                "graph_edge_drop_rate": graph.get("graph_edge_drop_rate", 0.0),
                "risk_center_noise": graph.get("risk_center_noise", 0.0),
                "risk_radius_scale": graph.get("risk_radius_scale", 1.0),
            }
        )

    n = max(len(cases), 1)
    summary = {
        "run_label": args.run_label,
        "planner_mode": args.planner_mode,
        "algo": args.algo,
        "map_id": str(cases[0].get("map_id", "")) if cases else "",
        "seed": args.seed,
        "episodes": len(cases),
        "success_rate": successes / n,
        "collision_rate": collisions / n,
        "timeout_rate": timeouts / n,
        "invalid_route_rate": invalid_routes / n,
        "repaired_route_rate": repaired_routes / n,
        "strict_llm_plan_valid_rate": strict_valid / n,
        "parse_ok_rate": parse_ok_count / n,
        "retry_success_rate": retry_successes / n,
        "no_llm_route_rate": no_llm_routes / n,
        "raw_llm_execution_rate": raw_execution_count / n,
        "mean_reward": total_reward / n,
        "mean_steps_all": sum(steps_all) / max(len(steps_all), 1),
        "mean_steps_success": sum(steps_success) / max(len(steps_success), 1),
        "mean_route_distance": sum(route_lengths) / max(len(route_lengths), 1),
        "mean_route_turns": sum(route_turns) / max(len(route_turns), 1),
        "mean_route_execution_score": sum(route_scores) / max(len(route_scores), 1),
        "mean_trajectory_semantic_cost": sum(semantic_costs) / max(len(semantic_costs), 1),
        "mean_edit_distance": sum(edit_distances) / max(len(edit_distances), 1),
        "mean_node_retention": sum(retained_fracs) / max(len(retained_fracs), 1),
        "safe_success_rate": successes / n if cases and str(cases[0].get("scenario")) == "semantic_constraint" else 0.0,
        "graph_edge_drop_rate": args.graph_edge_drop_rate,
        "risk_center_noise": args.risk_center_noise,
        "risk_radius_scale": args.risk_radius_scale,
        "order_gate_step_rate": order_gate_steps_total / n,
        "order_gate_accept_rate": order_gate_accepts_total / max(order_gate_steps_total, 1),
        "order_gate_fallback_rate": order_gate_fallbacks_total / max(order_gate_steps_total, 1),
        "mean_order_gate_consistency": sum(order_gate_consistency_values) / max(len(order_gate_consistency_values), 1),
        "mean_order_gate_vote_entropy": sum(order_gate_entropy_values) / max(len(order_gate_entropy_values), 1),
    }
    for key, value in sorted(failure_counts.items()):
        summary[f"failure_{key}_rate"] = value / n
    print("--- overall summary ---")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    if args.episode_csv:
        args.episode_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(episode_rows[0]) if episode_rows else [
            "run_label",
            "planner_mode",
            "algo",
            "case_id",
            "episode",
            "map_id",
            "seed",
            "scenario",
            "outcome",
        ]
        with args.episode_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"episode_csv: {args.episode_csv}")
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        print(f"summary_csv: {args.summary_csv}")


if __name__ == "__main__":
    main()
