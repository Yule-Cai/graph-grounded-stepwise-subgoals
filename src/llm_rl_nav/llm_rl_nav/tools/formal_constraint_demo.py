from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_rl_nav.formal_pipeline import SceneGraph, compile_rule


def default_semantic_map_path() -> Path:
    root = Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()
    return root / "src" / "llm_rl_nav" / "config" / "semantic_maps" / "hospital_semantic.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean AAAI formal constraint pipeline.")
    parser.add_argument("--text", "-t", default="晚上不要进入卧室")
    parser.add_argument("--semantic-map", default=str(default_semantic_map_path()))
    parser.add_argument(
        "--goal-entity",
        default=None,
        help="Optional task goal entity id for conflict validation.",
    )
    args = parser.parse_args()

    scene_graph = SceneGraph.from_file(args.semantic_map)
    result = compile_rule(args.text, scene_graph, goal_entity_id=args.goal_entity)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
