from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class Observation(BaseModel):
    soil_moisture: List[float] = Field(default_factory=list)
    water_budget: float = 0.0
    rain_forecast: float = 0.0
    temperature: float = 0.0
    day: int = 0
    max_steps: int = 0
    target_moisture: float = 55.0
    last_reward: float = 0.0
    last_action: Dict[str, object] = Field(default_factory=dict)
    last_reasoning: str = ""


class Action(BaseModel):
    zone_id: int = Field(..., ge=0)
    water_mm: float = Field(..., ge=0.0, le=12.0)


class Reward(BaseModel):
    value: float
