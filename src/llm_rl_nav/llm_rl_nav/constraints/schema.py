from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_CONSTRAINT_TYPES = {
    "forbidden_zone",
    "min_distance",
    "speed_limit_near",
    "prefer_region",
    "goal_region",
}


@dataclass(frozen=True)
class Constraint:
    """Machine-interpretable rule produced from a user's natural-language rule."""

    type: str
    target: str
    severity: str = "hard"
    distance_m: float | None = None
    speed_mps: float | None = None
    source_text: str | None = None
    confidence: float = 1.0
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "target": self.target,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
        }
        if self.distance_m is not None:
            data["distance_m"] = self.distance_m
        if self.speed_mps is not None:
            data["speed_mps"] = self.speed_mps
        if self.source_text:
            data["source_text"] = self.source_text
        if self.rationale:
            data["rationale"] = self.rationale
        return data


@dataclass
class ConstraintSet:
    constraints: list[Constraint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "warnings": self.warnings,
            "unknown_phrases": self.unknown_phrases,
        }
