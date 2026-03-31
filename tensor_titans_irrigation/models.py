from __future__ import annotations

from typing import Dict, List

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:  # pragma: no cover
    from pydantic import BaseModel

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        reward: float | None = None
        done: bool = False
        metadata: Dict[str, object] = Field(default_factory=dict)


class TensorTitansIrrigationAction(Action):
    zone_id: int = Field(..., ge=0, description="Zone index to irrigate.")
    water_mm: float = Field(..., ge=0.0, le=12.0, description="Requested irrigation in mm.")


class TensorTitansIrrigationObservation(Observation):
    soil_moisture: List[float] = Field(default_factory=list, description="Current soil moisture per zone.")
    water_budget: float = Field(default=0.0, description="Remaining water budget.")
    rain_forecast: float = Field(default=0.0, description="Rain forecast percentage.")
    temperature: float = Field(default=0.0, description="Current ambient temperature.")
    day: int = Field(default=0, description="Current day within the episode.")
    target_moisture: float = Field(default=55.0, description="Task moisture target.")
    last_reasoning: str = Field(default="", description="Structured text summary for the latest step.")
