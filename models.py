from pydantic import BaseModel
from typing import List


class Observation(BaseModel):
    soil_moisture: List[float]
    water_budget: float


class Action(BaseModel):
    zone_id: int
    water_mm: float


class Reward(BaseModel):
    value: float