from __future__ import annotations

from dataclasses import dataclass

from .schema import Constraint
from .semantic_map import SemanticMap


@dataclass(frozen=True)
class PathViolation:
    constraint_type: str
    target: str
    point: tuple[float, float]
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_type": self.constraint_type,
            "target": self.target,
            "point": [self.point[0], self.point[1]],
            "message": self.message,
        }


def check_path(
    semantic_map: SemanticMap,
    constraints: list[Constraint],
    path: list[tuple[float, float]],
) -> list[PathViolation]:
    violations: list[PathViolation] = []
    for point in path:
        x, y = point
        for constraint in constraints:
            if constraint.type == "forbidden_zone":
                if semantic_map.contains_point(constraint.target, x, y):
                    violations.append(
                        PathViolation(
                            constraint_type=constraint.type,
                            target=constraint.target,
                            point=point,
                            message=f"path enters forbidden zone {constraint.target}",
                        )
                    )
            elif constraint.type == "min_distance":
                min_distance = constraint.distance_m or 1.0
                distance = semantic_map.distance_to_entity(constraint.target, x, y)
                if distance < min_distance:
                    violations.append(
                        PathViolation(
                            constraint_type=constraint.type,
                            target=constraint.target,
                            point=point,
                            message=(
                                f"path is {distance:.2f}m from {constraint.target}, "
                                f"below {min_distance:.2f}m"
                            ),
                        )
                    )
    return violations


def parse_path(text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for raw_point in text.split(";"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        x_text, y_text = raw_point.split(",", maxsplit=1)
        points.append((float(x_text), float(y_text)))
    return points
