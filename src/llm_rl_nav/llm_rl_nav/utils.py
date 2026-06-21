from __future__ import annotations

import math
import os
from pathlib import Path


LATEST_SUCCESSFUL_PPO_MODEL = "ppo_reference_family_flat_goalfirst_5000000_from_scratch.zip"


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def latest_successful_ppo_path(root: Path | None = None) -> Path:
    base = root or project_root()
    return base / "models" / "PPO" / LATEST_SUCCESSFUL_PPO_MODEL
