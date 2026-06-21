from __future__ import annotations

import re
from collections.abc import Iterable

from .schema import Constraint, ConstraintSet
from .semantic_map import SemanticMap


CLAUSE_SPLIT_RE = re.compile(r"[。！？!?；;，,\n]+")
DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米|公尺)")
SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m/s|米每秒|米/秒)")

FORBIDDEN_MARKERS = (
    "不要进",
    "别进",
    "不能进",
    "禁止进入",
    "不要进入",
    "千万别进",
    "不能去",
    "别去",
    "不要去",
    "do not enter",
    "don't enter",
    "never enter",
    "avoid entering",
    "stay out of",
)

KEEP_DISTANCE_MARKERS = (
    "远离",
    "离",
    "不要靠近",
    "别靠近",
    "保持距离",
    "keep away",
    "stay away",
    "keep distance",
    "do not get close",
)

SLOW_MARKERS = (
    "慢一点",
    "慢点",
    "减速",
    "放慢",
    "slow down",
    "go slowly",
)

PREFER_MARKERS = (
    "优先走",
    "尽量走",
    "可以走",
    "走主",
    "prefer",
    "use the",
)

GOAL_MARKERS = (
    "去",
    "到",
    "前往",
    "导航到",
    "go to",
    "navigate to",
)


class NaturalLanguageConstraintCompiler:
    """Deterministic baseline for natural-language-to-constraint translation.

    The ACL-facing system can later swap this class for an LLM backend while
    keeping the same machine-readable constraint schema.
    """

    def __init__(self, semantic_map: SemanticMap):
        self.semantic_map = semantic_map

    def compile(self, text: str) -> ConstraintSet:
        constraints: list[Constraint] = []
        warnings: list[str] = []
        unknown: list[str] = []

        clauses = [clause.strip() for clause in CLAUSE_SPLIT_RE.split(text) if clause.strip()]
        for clause in clauses:
            matched = self._compile_clause(clause)
            if matched:
                constraints.extend(matched)
            else:
                unknown.append(clause)

        constraints = self._dedupe(constraints)
        if not constraints:
            warnings.append("No executable constraint could be inferred from the input.")
        return ConstraintSet(constraints=constraints, warnings=warnings, unknown_phrases=unknown)

    def _compile_clause(self, clause: str) -> list[Constraint]:
        lower = clause.lower()
        constraints: list[Constraint] = []

        if _contains_any(lower, FORBIDDEN_MARKERS):
            constraints.extend(self._constraints_for_entities(clause, "forbidden_zone", "hard"))

        if _contains_any(lower, KEEP_DISTANCE_MARKERS):
            distance = _extract_distance(lower)
            for constraint in self._constraints_for_entities(clause, "min_distance", "hard"):
                target = self.semantic_map.entities[constraint.target]
                constraints.append(
                    Constraint(
                        type=constraint.type,
                        target=constraint.target,
                        severity=constraint.severity,
                        distance_m=distance or target.default_min_distance or 1.0,
                        source_text=clause,
                        confidence=constraint.confidence,
                        rationale="keep-distance phrase mapped to a metric distance constraint",
                    )
                )

        if _contains_any(lower, SLOW_MARKERS):
            speed = _extract_speed(lower) or 0.2
            for constraint in self._constraints_for_entities(clause, "speed_limit_near", "soft"):
                constraints.append(
                    Constraint(
                        type=constraint.type,
                        target=constraint.target,
                        severity=constraint.severity,
                        speed_mps=speed,
                        source_text=clause,
                        confidence=constraint.confidence,
                        rationale="slow-down phrase mapped to local speed limit",
                    )
                )

        if _contains_any(lower, PREFER_MARKERS):
            constraints.extend(self._constraints_for_entities(clause, "prefer_region", "soft"))

        if _contains_any(lower, GOAL_MARKERS) and not constraints:
            constraints.extend(self._constraints_for_entities(clause, "goal_region", "soft"))

        return constraints

    def _constraints_for_entities(
        self,
        clause: str,
        constraint_type: str,
        severity: str,
    ) -> list[Constraint]:
        entities = self._find_entities(clause)
        return [
            Constraint(
                type=constraint_type,
                target=entity_id,
                severity=severity,
                source_text=clause,
                confidence=confidence,
                rationale=f"matched semantic map alias for {entity_id}",
            )
            for entity_id, confidence in entities
        ]

    def _find_entities(self, clause: str) -> list[tuple[str, float]]:
        normalized_clause = _normalize(clause)
        matches: list[tuple[int, str, float]] = []
        for alias, entity_id in self.semantic_map.alias_entries:
            if not alias:
                continue
            if alias in normalized_clause:
                confidence = min(0.95, 0.55 + len(alias) / max(len(normalized_clause), 1))
                matches.append((len(alias), entity_id, confidence))

        if not matches:
            entity = self.semantic_map.resolve(clause)
            if entity:
                return [(entity.entity_id, 0.5)]
            return []

        deduped: dict[str, tuple[int, float]] = {}
        for length, entity_id, confidence in matches:
            previous = deduped.get(entity_id)
            if previous is None or length > previous[0]:
                deduped[entity_id] = (length, confidence)
        return [
            (entity_id, confidence)
            for entity_id, (_, confidence) in sorted(
                deduped.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )
        ]

    @staticmethod
    def _dedupe(constraints: Iterable[Constraint]) -> list[Constraint]:
        seen: set[tuple[str, str, float | None, float | None]] = set()
        deduped: list[Constraint] = []
        for constraint in constraints:
            key = (
                constraint.type,
                constraint.target,
                constraint.distance_m,
                constraint.speed_mps,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(constraint)
        return deduped


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _extract_distance(text: str) -> float | None:
    match = DISTANCE_RE.search(text)
    if match:
        return float(match.group(1))
    if "远一点" in text or "远点" in text:
        return 1.0
    if "很远" in text or "far away" in text:
        return 2.0
    return None


def _extract_speed(text: str) -> float | None:
    match = SPEED_RE.search(text)
    if match:
        return float(match.group(1))
    return None


def _normalize(text: str) -> str:
    return "".join(text.lower().replace("_", " ").split())
