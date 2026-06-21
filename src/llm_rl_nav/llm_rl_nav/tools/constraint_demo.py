from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_rl_nav.constraints import (
    LMStudioConstraintCompiler,
    NaturalLanguageConstraintCompiler,
    SemanticMap,
)
from llm_rl_nav.constraints.path_checker import check_path, parse_path
from llm_rl_nav.constraints.validator import ConstraintValidator


def default_semantic_map_path() -> Path:
    project_root = Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()
    return project_root / "src" / "llm_rl_nav" / "config" / "semantic_maps" / "hospital_semantic.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile non-expert natural-language rules into robot constraints."
    )
    parser.add_argument(
        "--text",
        "-t",
        default="不要进红色厨房区，离蓝色花瓶远一点，可以走主走廊。",
        help="Natural-language rule to compile.",
    )
    parser.add_argument(
        "--semantic-map",
        default=str(default_semantic_map_path()),
        help="Path to semantic map YAML/JSON.",
    )
    parser.add_argument(
        "--path",
        default="0,-25;-11,-26;10.8,3.2;0,25",
        help="Optional path points as 'x,y;x,y' for violation checking.",
    )
    parser.add_argument(
        "--compiler",
        choices=["local", "lmstudio"],
        default="local",
        help="Compiler backend. local is deterministic; lmstudio calls an OpenAI-compatible LM Studio server.",
    )
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default=os.environ.get("LM_STUDIO_MODEL", "local-model"))
    args = parser.parse_args()

    semantic_map = SemanticMap.from_file(args.semantic_map)
    if args.compiler == "lmstudio":
        compiler = LMStudioConstraintCompiler(
            semantic_map,
            base_url=args.lm_studio_url,
            model=args.model,
        )
    else:
        compiler = NaturalLanguageConstraintCompiler(semantic_map)
    validator = ConstraintValidator(semantic_map)

    compiled = compiler.compile(args.text)
    validated = validator.validate(compiled)

    result = validated.to_dict()
    if args.path:
        path = parse_path(args.path)
        result["path"] = [[x, y] for x, y in path]
        result["path_violations"] = [
            violation.to_dict()
            for violation in check_path(semantic_map, validated.constraints, path)
        ]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
