from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import yaml
except ImportError:  # pragma: no cover - ROS environments usually provide PyYAML.
    yaml = None


@dataclass(frozen=True)
class SemanticEntity:
    entity_id: str
    entity_kind: str
    semantic_type: str
    shape: str
    center: tuple[float, float] | None = None
    size: tuple[float, float] | None = None
    radius: float | None = None
    position: tuple[float, float] | None = None
    aliases: tuple[str, ...] = ()
    default_min_distance: float | None = None

    @property
    def point(self) -> tuple[float, float] | None:
        return self.position or self.center


class SemanticMap:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.entities = self._load_entities(data)
        self.alias_entries = self._build_alias_entries(self.entities)
        self.alias_index = self._build_alias_index(self.alias_entries)

    @classmethod
    def from_file(cls, path: str | Path) -> "SemanticMap":
        map_path = Path(path)
        text = map_path.read_text(encoding="utf-8")
        if map_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load semantic YAML maps.")
            data = yaml.safe_load(text)
        return cls(data)

    def resolve(self, phrase: str) -> SemanticEntity | None:
        normalized = _normalize(phrase)
        if normalized in self.alias_index:
            return self.entities[self.alias_index[normalized]]

        candidates: list[tuple[int, str]] = []
        for alias, entity_id in self.alias_entries:
            if alias and alias in normalized:
                candidates.append((len(alias), entity_id))
            elif normalized and normalized in alias:
                candidates.append((len(normalized), entity_id))
        if not candidates:
            return None

        _, entity_id = max(candidates, key=lambda item: item[0])
        return self.entities[entity_id]

    def all_aliases(self) -> dict[str, list[str]]:
        return {
            entity_id: list(entity.aliases)
            for entity_id, entity in sorted(self.entities.items())
        }

    def contains_point(self, entity_id: str, x: float, y: float) -> bool:
        entity = self.entities[entity_id]
        if entity.shape == "rect" and entity.center and entity.size:
            cx, cy = entity.center
            sx, sy = entity.size
            return abs(x - cx) <= sx / 2 and abs(y - cy) <= sy / 2
        if entity.shape == "circle" and entity.center and entity.radius is not None:
            return _distance((x, y), entity.center) <= entity.radius
        return False

    def distance_to_entity(self, entity_id: str, x: float, y: float) -> float:
        entity = self.entities[entity_id]
        if entity.shape == "rect" and entity.center and entity.size:
            return _distance_to_rect((x, y), entity.center, entity.size)
        if entity.shape == "circle" and entity.center and entity.radius is not None:
            return max(0.0, _distance((x, y), entity.center) - entity.radius)
        point = entity.point
        if point is None:
            return math.inf
        return _distance((x, y), point)

    @staticmethod
    def _load_entities(data: dict[str, Any]) -> dict[str, SemanticEntity]:
        entities: dict[str, SemanticEntity] = {}
        for region_id, region in data.get("regions", {}).items():
            center = _as_point(region.get("center"))
            size = _as_point(region.get("size"))
            entities[region_id] = SemanticEntity(
                entity_id=region_id,
                entity_kind="region",
                semantic_type=region.get("type", "region"),
                shape=region.get("shape", "rect"),
                center=center,
                size=size,
                radius=_as_float(region.get("radius")),
                aliases=tuple(region.get("aliases", ())),
                default_min_distance=_as_float(region.get("default_min_distance")),
            )
        for object_id, obj in data.get("objects", {}).items():
            position = _as_point(obj.get("position"))
            size = _as_point(obj.get("size"))
            radius = _as_float(obj.get("radius"))
            shape = obj.get("shape", "circle" if radius is not None else "point")
            entities[object_id] = SemanticEntity(
                entity_id=object_id,
                entity_kind="object",
                semantic_type=obj.get("type", "object"),
                shape=shape,
                center=position if shape in {"circle", "rect"} else None,
                size=size,
                radius=radius,
                position=position,
                aliases=tuple(obj.get("aliases", ())),
                default_min_distance=_as_float(obj.get("default_min_distance")),
            )
        return entities

    @staticmethod
    def _build_alias_entries(entities: dict[str, SemanticEntity]) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for entity_id, entity in entities.items():
            names = [entity_id, entity_id.replace("_", " "), *entity.aliases]
            for name in names:
                entries.append((_normalize(name), entity_id))
        return entries

    @staticmethod
    def _build_alias_index(alias_entries: list[tuple[str, str]]) -> dict[str, str]:
        index: dict[str, str] = {}
        for alias, entity_id in alias_entries:
            index.setdefault(alias, entity_id)
        return index


def _normalize(text: str) -> str:
    return "".join(str(text).lower().replace("_", " ").split())


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_point(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) < 2:
        return None
    return (float(value[0]), float(value[1]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _distance_to_rect(
    point: tuple[float, float],
    center: tuple[float, float],
    size: tuple[float, float],
) -> float:
    dx = max(abs(point[0] - center[0]) - size[0] / 2, 0.0)
    dy = max(abs(point[1] - center[1]) - size[1] / 2, 0.0)
    return math.hypot(dx, dy)
