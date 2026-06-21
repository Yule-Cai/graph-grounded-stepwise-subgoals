from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .schema import Constraint, ConstraintSet
from .semantic_map import SemanticMap


class LMStudioConstraintCompiler:
    """Compile natural-language rules with an OpenAI-compatible LM Studio server."""

    def __init__(
        self,
        semantic_map: SemanticMap,
        base_url: str = "http://localhost:1234/v1",
        model: str | None = None,
    ):
        self.semantic_map = semantic_map
        self.base_url = base_url
        self.model = model or os.environ.get("LM_STUDIO_MODEL", "local-model")
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")

    def compile(self, text: str) -> ConstraintSet:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        content = response.choices[0].message.content or ""
        data = _loads_json(content)
        constraints: list[Constraint] = []
        warnings = list(data.get("warnings", []))
        for item in _constraint_items(data):
            try:
                constraints.extend(_constraints_from_dict(item, text, self.semantic_map))
            except ValueError as exc:
                warnings.append(f"Skipped malformed LM Studio constraint {item}: {exc}")
        unknown = list(data.get("unknown_phrases", []))
        warnings.append(f"LM Studio raw response: {content}")
        return ConstraintSet(constraints=constraints, warnings=warnings, unknown_phrases=unknown)

    def _system_prompt(self) -> str:
        aliases = self.semantic_map.all_aliases()
        return f"""
You are a compiler from non-expert natural-language robot rules to executable JSON constraints.

Return only valid JSON with this schema:
{{
  "constraints": [
    {{
      "type": "forbidden_zone | min_distance | speed_limit_near | prefer_region | goal_region",
      "target": "one semantic target id",
      "severity": "hard | soft",
      "distance_m": optional number,
      "speed_mps": optional number
    }}
  ],
  "warnings": [],
  "unknown_phrases": []
}}

Supported targets and aliases:
{json.dumps(aliases, ensure_ascii=False, indent=2)}

Rules:
- Map "不要进入/别进/禁止进入/不准去" to forbidden_zone.
- Map "离...远一点/不要靠近/保持距离" to min_distance. Use distance_m=1.0 unless the user gives a number.
- Map "慢一点/减速" near a target to speed_limit_near with speed_mps=0.2.
- If the user says 卧室 or bedroom, target both west_patient_rooms and east_patient_rooms.
- If the user says 警区, 警戒区, 禁区, 厨房, or red zone, use red_no_entry_kitchen.
- If the user says 古董花瓶, 花瓶, 易碎物, or vase, use blue_fragile_vase_zone.
- Do not invent targets that are not in the supported target list.
- Return JSON only. No markdown.
""".strip()


def _loads_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last >= first:
        cleaned = cleaned[first : last + 1]
    return json.loads(cleaned)


def _constraint_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("constraints"), list):
        return [item for item in data["constraints"] if isinstance(item, dict)]
    if isinstance(data.get("constraint"), dict):
        return [data["constraint"]]
    if any(key in data for key in ("type", "constraint_type", "action", "target", "zone", "room")):
        return [data]
    return []


def _constraints_from_dict(
    data: dict[str, Any],
    source_text: str,
    semantic_map: SemanticMap,
) -> list[Constraint]:
    constraint_type = _normalize_type(
        data.get("type")
        or data.get("constraint_type")
        or data.get("action")
        or data.get("rule")
        or data.get("constraint")
    )
    targets = _normalize_targets(
        data.get("target") or data.get("zone") or data.get("room") or data.get("object"),
        semantic_map,
    )
    if constraint_type is None:
        raise ValueError("missing or unsupported type")
    if not targets:
        raise ValueError("missing or unknown target")

    return [
        Constraint(
            type=constraint_type,
            target=target,
            severity=str(data.get("severity", "hard")),
            distance_m=_optional_float(data.get("distance_m")),
            speed_mps=_optional_float(data.get("speed_mps")),
            source_text=source_text,
            confidence=float(data.get("confidence", 1.0)),
            rationale=data.get("rationale", "compiled by LM Studio"),
        )
        for target in targets
    ]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"forbidden_zone", "no_entry", "avoid", "avoid_zone", "prohibit", "prohibited"}:
        return "forbidden_zone"
    if any(marker in text for marker in ("不要进入", "别进", "禁止进入", "不准去", "不要进", "不能进")):
        return "forbidden_zone"
    if any(marker in text for marker in ("远离", "保持距离", "不要靠近", "离")):
        return "min_distance"
    if any(marker in text for marker in ("慢一点", "慢点", "减速")):
        return "speed_limit_near"
    if text in {"min_distance", "keep_distance", "stay_away"}:
        return "min_distance"
    if text in {"speed_limit_near", "slow", "slow_down"}:
        return "speed_limit_near"
    if text in {"prefer_region", "prefer"}:
        return "prefer_region"
    if text in {"goal_region", "goal", "navigate_to"}:
        return "goal_region"
    return None


def _normalize_targets(value: Any, semantic_map: SemanticMap) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    normalized = "".join(text.lower().replace("_", " ").split())
    if normalized in {"卧室", "睡房", "bedroom", "room", "patientroom", "ward", "病房"}:
        return ["west_patient_rooms", "east_patient_rooms"]
    if normalized in {"警区", "警戒区", "禁区", "红色警戒区", "红色厨房区", "厨房", "redzone", "kitchen"}:
        return ["red_no_entry_kitchen"]
    if normalized in {"花瓶", "古董花瓶", "蓝色花瓶", "易碎物", "vase", "fragilevase"}:
        return ["blue_fragile_vase_zone"]
    if text in semantic_map.entities:
        return [text]
    entity = semantic_map.resolve(text)
    if entity:
        return [entity.entity_id]
    return []
