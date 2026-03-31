from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TensorTitansIrrigationAction, TensorTitansIrrigationObservation


class TensorTitansIrrigationEnv(
    EnvClient[TensorTitansIrrigationAction, TensorTitansIrrigationObservation, State]
):
    def _step_payload(self, action: TensorTitansIrrigationAction) -> Dict:
        return {
            "zone_id": action.zone_id,
            "water_mm": action.water_mm,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TensorTitansIrrigationObservation]:
        obs_data = payload.get("observation", {})
        observation = TensorTitansIrrigationObservation(
            soil_moisture=obs_data.get("soil_moisture", []),
            water_budget=obs_data.get("water_budget", 0.0),
            rain_forecast=obs_data.get("rain_forecast", 0.0),
            temperature=obs_data.get("temperature", 0.0),
            day=obs_data.get("day", 0),
            target_moisture=obs_data.get("target_moisture", 55.0),
            last_reasoning=obs_data.get("last_reasoning", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
