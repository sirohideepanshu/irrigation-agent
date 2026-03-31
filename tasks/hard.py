from __future__ import annotations

from typing import Optional

from irrigation_env import IrrigationEnv


def create_env(seed: Optional[int] = None) -> IrrigationEnv:
    return IrrigationEnv("hard", seed=seed)
