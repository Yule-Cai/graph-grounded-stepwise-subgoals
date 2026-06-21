from __future__ import annotations

from .schema import Constraint, ConstraintSet, SUPPORTED_CONSTRAINT_TYPES
from .semantic_map import SemanticMap


class ConstraintValidator:
    def __init__(self, semantic_map: SemanticMap):
        self.semantic_map = semantic_map

    def validate(self, constraint_set: ConstraintSet) -> ConstraintSet:
        valid: list[Constraint] = []
        warnings = list(constraint_set.warnings)

        for constraint in constraint_set.constraints:
            if constraint.type not in SUPPORTED_CONSTRAINT_TYPES:
                warnings.append(f"Unsupported constraint type: {constraint.type}")
                continue
            if constraint.target not in self.semantic_map.entities:
                warnings.append(f"Unknown target in semantic map: {constraint.target}")
                continue
            if constraint.type == "min_distance" and constraint.distance_m is None:
                entity = self.semantic_map.entities[constraint.target]
                valid.append(
                    Constraint(
                        type=constraint.type,
                        target=constraint.target,
                        severity=constraint.severity,
                        distance_m=entity.default_min_distance or 1.0,
                        source_text=constraint.source_text,
                        confidence=constraint.confidence,
                        rationale=constraint.rationale,
                    )
                )
                continue
            if constraint.type == "speed_limit_near" and constraint.speed_mps is None:
                valid.append(
                    Constraint(
                        type=constraint.type,
                        target=constraint.target,
                        severity=constraint.severity,
                        speed_mps=0.2,
                        source_text=constraint.source_text,
                        confidence=constraint.confidence,
                        rationale=constraint.rationale,
                    )
                )
                continue
            valid.append(constraint)

        return ConstraintSet(
            constraints=valid,
            warnings=warnings,
            unknown_phrases=constraint_set.unknown_phrases,
        )
