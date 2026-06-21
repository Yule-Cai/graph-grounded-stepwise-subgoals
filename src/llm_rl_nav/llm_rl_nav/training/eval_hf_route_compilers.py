from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from llm_rl_nav.envs.semantic_world_source import MAP_SOURCE_CHOICES, build_nav_env
from llm_rl_nav.training.eval_llm_route_planning import _candidate_route, _json_from_text, _normalize_indices
from llm_rl_nav.training.train_multimap_ppo import parse_maps


CAUSAL_ARCH_HINTS = ("ForCausalLM", "LMHeadModel")


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def _model_dirs(root: Path, selected: str | None = None) -> list[Path]:
    base = root / "external_models" / "hf"
    if selected:
        names = [item.strip() for item in selected.split(",") if item.strip()]
        return [base / name for name in names]
    return sorted(path for path in base.iterdir() if path.is_dir())


def _config(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def _is_generative(config: dict[str, Any]) -> bool:
    architectures = config.get("architectures") or []
    return any(any(hint in str(arch) for hint in CAUSAL_ARCH_HINTS) for arch in architectures)


def _params_label(config: dict[str, Any], model_dir: Path) -> str:
    if "num_parameters" in config:
        value = float(config["num_parameters"])
    else:
        value = 0.0
        for file in model_dir.glob("*.safetensors"):
            value += file.stat().st_size / 2.0
        for file in model_dir.glob("pytorch_model*.bin"):
            value += file.stat().st_size / 4.0
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    return "unknown"


def _safe_name(model_dir: Path) -> str:
    return model_dir.name.replace("__", "/")


def _prompt(map_id: str, robot: tuple[float, float], goal: tuple[float, float], candidates: list[tuple[float, float]]) -> str:
    nodes = [{"id": i, "x": x, "y": y} for i, (x, y) in enumerate(candidates)]
    edges = [[i, i + 1] for i in range(max(0, len(candidates) - 1))]
    payload = {
        "task": "Return JSON only.",
        "schema": {"route": [0, 1, 2], "warnings": []},
        "map_id": map_id,
        "robot_pose": {"x": round(robot[0], 2), "y": round(robot[1], 2)},
        "final_goal": {"x": round(goal[0], 2), "y": round(goal[1], 2)},
        "start_node_id": 0,
        "final_node_id": len(candidates) - 1,
        "navigation_graph": {"nodes": nodes, "edges": edges},
        "rules": [
            "Choose only node ids from navigation_graph.nodes.",
            "The route must start with start_node_id.",
            "The route must end with final_node_id.",
            "A partial prefix is invalid.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _parse_route(text: str, candidate_count: int) -> tuple[list[int], bool, bool, bool]:
    parse_ok = True
    try:
        parsed = _json_from_text(text)
        indices = _normalize_indices(parsed)
    except Exception:
        parse_ok = False
        indices = []
        for match in re.finditer(r"-?\d+", text):
            value = int(match.group(0))
            if value not in indices:
                indices.append(value)
            if len(indices) >= candidate_count:
                break

    valid_ids = [idx for idx in indices if 0 <= idx < candidate_count]
    all_ids_valid = len(valid_ids) == len(indices) and bool(valid_ids)
    starts_ok = bool(valid_ids) and valid_ids[0] == 0
    final_ok = bool(valid_ids) and valid_ids[-1] == candidate_count - 1
    monotonic_edges = all(b >= a for a, b in zip(valid_ids, valid_ids[1:]))
    complete_route = parse_ok and all_ids_valid and starts_ok and final_ok and monotonic_edges
    return valid_ids, parse_ok, final_ok, complete_route


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1400)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    root = project_root()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="Evaluate local Hugging Face models as route compilers.")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--maps", type=parse_maps, default=("reference_family_flat",))
    parser.add_argument("--map-source", default="gazebo_3d_projection", choices=MAP_SOURCE_CHOICES)
    parser.add_argument("--models", default=None, help="Comma-separated external_models/hf directory names.")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers and torch are required for HF model-zoo evaluation.") from exc

    if args.device == "mps":
        device = "mps"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    out_dir = root / "logs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"hf_route_compilers_{args.episodes}ep_{stamp}.csv"
    log_path = out_dir / f"hf_route_compilers_{args.episodes}ep_{stamp}.log"

    rows: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log:
        for model_dir in _model_dirs(root, args.models):
            if not (model_dir / "config.json").exists():
                continue
            config = _config(model_dir)
            model_name = _safe_name(model_dir)
            model_type = config.get("model_type", "unknown")
            generative = _is_generative(config)
            params = _params_label(config, model_dir)
            log.write(f"=== model={model_name} type={model_type} params={params} generative={int(generative)} ===\n")
            log.flush()

            total = 0
            parse_ok = 0
            final_ok = 0
            complete = 0
            avg_selected = 0.0
            avg_latency = 0.0
            load_error = ""

            model = None
            tokenizer = None
            if generative:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        local_files_only=True,
                        torch_dtype=torch.float16 if device == "mps" else torch.float32,
                        low_cpu_mem_usage=True,
                    )
                    model.to(device)
                    model.eval()
                except Exception as exc:
                    load_error = str(exc)
                    generative = False
                    log.write(f"load_error={load_error}\n")

            for map_index, map_id in enumerate(args.maps):
                for episode in range(args.episodes):
                    env = build_nav_env(
                        args.map_source,
                        seed=args.seed + map_index,
                        map_id=map_id,
                        max_steps=700,
                        reward_profile="v8_goal",
                        goal_min_distance=2.0,
                        goal_max_distance=12.0,
                        goal_point_probability=0.95,
                    )
                    reset = env.reset(seed=args.seed + map_index * 1000 + episode, options={"map_id": map_id})
                    if isinstance(reset, tuple):
                        _ = reset[0]
                    goal = (float(env.goal_x), float(env.goal_y))
                    robot = (float(env.robot_x), float(env.robot_y))
                    candidates = _candidate_route(env, goal, spacing=2.2, resolution=0.55)
                    if len(candidates) < 2:
                        continue
                    total += 1
                    if not generative or model is None or tokenizer is None:
                        log.write(f"episode={episode:03d} skipped_non_generative candidates={len(candidates)}\n")
                        continue
                    prompt = _prompt(map_id, robot, goal, candidates)
                    started = time.time()
                    try:
                        text = _generate(model, tokenizer, prompt, args.max_new_tokens)
                    except Exception as exc:
                        log.write(f"episode={episode:03d} generation_error={exc}\n")
                        continue
                    latency = time.time() - started
                    route, parsed, reached_final, valid = _parse_route(text, len(candidates))
                    parse_ok += int(parsed)
                    final_ok += int(reached_final)
                    complete += int(valid)
                    avg_selected += len(route)
                    avg_latency += latency
                    log.write(
                        f"episode={episode:03d} parse_ok={int(parsed)} final_ok={int(reached_final)} "
                        f"complete={int(valid)} selected={route[:24]} candidates={len(candidates)} "
                        f"latency={latency:.2f}s raw={text[:180].replace(chr(10), ' ')}\n"
                    )
                    log.flush()

            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            denom = max(total, 1)
            row = {
                "model": model_name,
                "model_type": model_type,
                "params": params,
                "generative": int(generative),
                "episodes": total,
                "json_parse_rate": parse_ok / denom,
                "final_node_rate": final_ok / denom,
                "complete_route_rate": complete / denom,
                "avg_selected_nodes": avg_selected / denom,
                "avg_latency_s": avg_latency / denom,
                "load_error": load_error[:180],
            }
            rows.append(row)
            print(
                f"{model_name}: parse={row['json_parse_rate']:.3f} "
                f"final={row['final_node_rate']:.3f} complete={row['complete_route_rate']:.3f} "
                f"latency={row['avg_latency_s']:.2f}s"
            )

    fieldnames = [
        "model",
        "model_type",
        "params",
        "generative",
        "episodes",
        "json_parse_rate",
        "final_node_rate",
        "complete_route_rate",
        "avg_selected_nodes",
        "avg_latency_s",
        "load_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
