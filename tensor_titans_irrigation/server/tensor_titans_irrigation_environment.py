from __future__ import annotations

from uuid import uuid4

from irrigation_env import IrrigationEnv

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:  # pragma: no cover
    from pydantic import BaseModel

    class State(BaseModel):
        episode_id: str
        step_count: int = 0

    class Environment:
        pass

try:
    from ..models import TensorTitansIrrigationAction, TensorTitansIrrigationObservation
except ImportError:  # pragma: no cover
    from tensor_titans_irrigation.models import TensorTitansIrrigationAction, TensorTitansIrrigationObservation


class TensorTitansIrrigationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._env = IrrigationEnv("hard", seed=2026)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> TensorTitansIrrigationObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        observation = self._env.reset(seed=2026)
        return self._to_observation(observation, reward=0.0, done=False, metadata={"task": "hard"})

    def step(self, action: TensorTitansIrrigationAction) -> TensorTitansIrrigationObservation:  # type: ignore[override]
        self._state.step_count += 1
        observation, reward, done, info = self._env.step({"zone_id": action.zone_id, "water_mm": action.water_mm})
        return self._to_observation(observation, reward=reward, done=done, metadata=info)

    @property
    def state(self) -> State:
        return self._state

    def _to_observation(self, observation: dict, reward: float, done: bool, metadata: dict) -> TensorTitansIrrigationObservation:
        explanation = dict(metadata.get("explanation", {}))
        return TensorTitansIrrigationObservation(
            soil_moisture=[float(value) for value in observation.get("soil_moisture", [])],
            water_budget=float(observation.get("water_budget", 0.0)),
            rain_forecast=float(observation.get("rain_forecast", 0.0)),
            temperature=float(observation.get("temperature", 0.0)),
            day=int(observation.get("day", 0)),
            target_moisture=float(observation.get("target_moisture", 55.0)),
            last_reasoning=str(explanation.get("summary", "")),
            reward=float(reward),
            done=bool(done),
            metadata=metadata,
        )
