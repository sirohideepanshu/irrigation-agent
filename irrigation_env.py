from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
import math

import numpy as np


SOIL_BOUNDS: Tuple[float, float] = (0.0, 100.0)
TEMPERATURE_BOUNDS: Tuple[float, float] = (-10.0, 60.0)
RAIN_FORECAST_BOUNDS: Tuple[float, float] = (0.0, 100.0)
MAX_WATER_MM = 12.0


@dataclass(frozen=True)
class TaskConfig:
    name: str
    zones: int
    max_steps: int
    water_budget: float
    target_moisture: float
    seed: int
    soil_range: Tuple[float, float]
    temperature_range: Tuple[float, float]
    rain_forecast_range: Tuple[float, float]
    evaporation_rate: float
    crop_uptake: float
    rain_capture: float


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        zones=3,
        max_steps=20,
        water_budget=90.0,
        target_moisture=52.0,
        seed=11,
        soil_range=(36.0, 58.0),
        temperature_range=(20.0, 30.0),
        rain_forecast_range=(10.0, 40.0),
        evaporation_rate=0.12,
        crop_uptake=1.1,
        rain_capture=0.60,
    ),
    "medium": TaskConfig(
        name="medium",
        zones=5,
        max_steps=30,
        water_budget=110.0,
        target_moisture=55.0,
        seed=23,
        soil_range=(24.0, 66.0),
        temperature_range=(22.0, 34.0),
        rain_forecast_range=(15.0, 65.0),
        evaporation_rate=0.16,
        crop_uptake=1.4,
        rain_capture=0.55,
    ),
    "hard": TaskConfig(
        name="hard",
        zones=7,
        max_steps=40,
        water_budget=150.0,
        target_moisture=58.0,
        seed=37,
        soil_range=(12.0, 76.0),
        temperature_range=(24.0, 39.0),
        rain_forecast_range=(10.0, 85.0),
        evaporation_rate=0.18,
        crop_uptake=1.5,
        rain_capture=0.55,
    ),
}


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _as_float(name: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")

    return numeric


def _soil_label(soil: float, target: float) -> str:
    if soil < target - 12:
        return "dry"
    if soil < target - 4:
        return "slightly dry"
    if soil <= target + 4:
        return "healthy"
    if soil <= target + 12:
        return "slightly wet"
    return "waterlogged"


def _weather_label(temp: float, rain_forecast: float, rainfall: float) -> str:
    rain_state = "dry outlook"
    if rain_forecast >= 70 or rainfall >= 5:
        rain_state = "rain-heavy outlook"
    elif rain_forecast >= 40 or rainfall >= 2:
        rain_state = "mixed outlook"

    if temp >= 35:
        return f"hot with a {rain_state}"
    if temp <= 18:
        return f"cool with a {rain_state}"
    return f"mild with a {rain_state}"


class IrrigationEnv:
    def __init__(self, config: TaskConfig | str = "medium", seed: Optional[int] = None):
        self.config = TASK_CONFIGS[config] if isinstance(config, str) else config
        self.seed = self.config.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.reset(seed=self.seed)

    def reset(
        self,
        seed: Optional[int] = None,
        initial_conditions: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)

        self._rng = np.random.default_rng(self.seed)
        self.day = 0
        self.water_budget = float(self.config.water_budget)
        self.water_used = 0.0
        self.last_reward = 0.0
        self.last_action: Dict[str, Any] = {}
        self.last_info: Dict[str, Any] = {}
        self.explanation_logs: List[Dict[str, Any]] = []

        self.soil_moisture = self._rng.uniform(
            self.config.soil_range[0],
            self.config.soil_range[1],
            self.config.zones,
        )
        self.temperature = float(
            self._rng.uniform(self.config.temperature_range[0], self.config.temperature_range[1])
        )
        self.rain_forecast = float(
            self._rng.uniform(
                self.config.rain_forecast_range[0],
                self.config.rain_forecast_range[1],
            )
        )
        self.last_rainfall = 0.0

        self._apply_initial_conditions(initial_conditions or {})
        return self.state()

    def state(self) -> Dict[str, Any]:
        return {
            "task_name": self.config.name,
            "soil_moisture": [float(round(value, 4)) for value in self.soil_moisture.tolist()],
            "water_budget": float(round(self.water_budget, 4)),
            "rain_forecast": float(round(self.rain_forecast, 4)),
            "temperature": float(round(self.temperature, 4)),
            "day": int(self.day),
            "max_steps": int(self.config.max_steps),
            "target_moisture": float(self.config.target_moisture),
            "last_reward": float(round(self.last_reward, 6)),
            "last_action": dict(self.last_action),
        }

    def step(self, action: Mapping[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        zone_id, requested_water, applied_water, adjustments = self._validate_action(action)
        soil_before = self.soil_moisture.copy()
        weather = self._simulate_weather()
        self._apply_soil_dynamics(zone_id, applied_water, weather)

        target = self._target_for_zone(zone_id)
        reward, reward_breakdown = self._calculate_reward(
            zone_id=zone_id,
            soil_before=float(soil_before[zone_id]),
            soil_after=float(self.soil_moisture[zone_id]),
            requested_water=requested_water,
            applied_water=applied_water,
            weather=weather,
            target=target,
        )

        self.day += 1
        self.last_reward = reward
        self.last_action = {
            "zone_id": int(zone_id),
            "requested_water_mm": float(round(requested_water, 4)),
            "applied_water_mm": float(round(applied_water, 4)),
            "validation_adjustments": adjustments,
        }

        explanation = self._build_explanation(
            zone_id=zone_id,
            soil_before=float(soil_before[zone_id]),
            soil_after=float(self.soil_moisture[zone_id]),
            applied_water=applied_water,
            target=target,
            weather=weather,
            reward_breakdown=reward_breakdown,
            adjustments=adjustments,
        )
        self.explanation_logs.append(explanation)

        self.rain_forecast = float(weather["next_rain_forecast"])
        self.temperature = float(weather["next_temperature"])
        done = self.day >= self.config.max_steps
        observation = self.state()

        info = {
            "weather": weather,
            "reward_breakdown": reward_breakdown,
            "explanation": explanation,
            "action_validation": adjustments,
            "recommended_water_mm": float(round(reward_breakdown["recommended_water_mm"], 4)),
        }
        self.last_info = info
        return observation, float(reward), bool(done), info

    def _apply_initial_conditions(self, initial_conditions: Mapping[str, Any]) -> None:
        if not initial_conditions:
            self._validate_observation()
            return

        selected_zone = int(initial_conditions.get("selected_zone", 0))
        selected_zone = max(0, min(self.config.zones - 1, selected_zone))

        soil_override = initial_conditions.get("soil_moisture")
        if isinstance(soil_override, list):
            if len(soil_override) != self.config.zones:
                raise ValueError("soil_moisture list must match the number of zones.")
            self.soil_moisture = np.array(
                [_clamp(_as_float("soil_moisture", value), SOIL_BOUNDS) for value in soil_override],
                dtype=float,
            )
        elif soil_override is not None:
            self.soil_moisture[selected_zone] = _clamp(
                _as_float("soil_moisture", soil_override),
                SOIL_BOUNDS,
            )

        if "temperature" in initial_conditions:
            self.temperature = _clamp(
                _as_float("temperature", initial_conditions["temperature"]),
                TEMPERATURE_BOUNDS,
            )

        if "rain_forecast" in initial_conditions:
            self.rain_forecast = _clamp(
                _as_float("rain_forecast", initial_conditions["rain_forecast"]),
                RAIN_FORECAST_BOUNDS,
            )

        self._validate_observation()

    def _validate_observation(self) -> None:
        self.soil_moisture = np.clip(self.soil_moisture.astype(float), SOIL_BOUNDS[0], SOIL_BOUNDS[1])
        self.temperature = _clamp(float(self.temperature), TEMPERATURE_BOUNDS)
        self.rain_forecast = _clamp(float(self.rain_forecast), RAIN_FORECAST_BOUNDS)
        self.water_budget = max(0.0, float(self.water_budget))

    def _validate_action(self, action: Mapping[str, Any]) -> Tuple[int, float, float, List[str]]:
        if not isinstance(action, Mapping):
            raise TypeError("Action must be a mapping with zone_id and water_mm.")

        if "zone_id" not in action or "water_mm" not in action:
            raise ValueError("Action must include zone_id and water_mm.")

        zone_id = int(action["zone_id"])
        if zone_id < 0 or zone_id >= self.config.zones:
            raise ValueError(f"zone_id must be between 0 and {self.config.zones - 1}.")

        requested_water = _as_float("water_mm", action["water_mm"])
        applied_water = requested_water
        adjustments: List[str] = []

        if applied_water < 0:
            adjustments.append("Negative irrigation request clipped to 0 mm.")
            applied_water = 0.0
        if applied_water > MAX_WATER_MM:
            adjustments.append(f"Irrigation request capped at {MAX_WATER_MM:.0f} mm.")
            applied_water = MAX_WATER_MM
        if applied_water > self.water_budget:
            adjustments.append("Irrigation request reduced to the remaining water budget.")
            applied_water = self.water_budget

        self.water_used += applied_water
        self.water_budget = max(0.0, self.water_budget - applied_water)
        return zone_id, requested_water, applied_water, adjustments

    def _simulate_weather(self) -> Dict[str, float]:
        rainfall_mean = (self.rain_forecast / 100.0) * (3.0 + self.config.rain_capture * 4.0)
        rainfall_noise = self._rng.normal(0.0, 1.1 + self.config.rain_capture)
        rainfall = _clamp(rainfall_mean + rainfall_noise, (0.0, 14.0))

        next_forecast = _clamp(
            0.62 * self.rain_forecast + rainfall * 6.5 + self._rng.normal(0.0, 10.0),
            RAIN_FORECAST_BOUNDS,
        )
        next_temperature = _clamp(
            self.temperature + self._rng.normal(0.0, 1.8),
            TEMPERATURE_BOUNDS,
        )
        self.last_rainfall = float(rainfall)

        return {
            "temperature": float(round(self.temperature, 4)),
            "rain_forecast": float(round(self.rain_forecast, 4)),
            "rainfall": float(round(rainfall, 4)),
            "next_temperature": float(round(next_temperature, 4)),
            "next_rain_forecast": float(round(next_forecast, 4)),
        }

    def _apply_soil_dynamics(self, zone_id: int, applied_water: float, weather: Mapping[str, float]) -> None:
        rainfall = float(weather["rainfall"])
        temperature = float(weather["temperature"])
        time_fraction = 1.0 / max(1, self.config.zones)

        for index in range(self.config.zones):
            zone_variation = 1.0 + 0.04 * ((index % 3) - 1)
            irrigation_gain = applied_water * 0.82 * zone_variation if index == zone_id else 0.0
            rainfall_gain = rainfall * self.config.rain_capture * (0.90 + 0.03 * index) * time_fraction
            evaporation = (
                max(0.0, temperature - 18.0)
                * self.config.evaporation_rate
                * (0.92 + 0.02 * index)
                * time_fraction
            )
            crop_use = self.config.crop_uptake * (0.95 + 0.03 * (index % 2)) * time_fraction
            drainage = max(0.0, self.soil_moisture[index] - 74.0) * 0.10
            overspray = max(0.0, applied_water - 8.0) * 0.20 if index == zone_id else 0.0

            updated = (
                self.soil_moisture[index]
                + irrigation_gain
                + rainfall_gain
                - evaporation
                - crop_use
                - drainage
                - overspray
            )
            self.soil_moisture[index] = _clamp(float(updated), SOIL_BOUNDS)

    def _calculate_reward(
        self,
        zone_id: int,
        soil_before: float,
        soil_after: float,
        requested_water: float,
        applied_water: float,
        weather: Mapping[str, float],
        target: float,
    ) -> Tuple[float, Dict[str, float]]:
        rain_forecast = float(weather["rain_forecast"])
        rainfall = float(weather["rainfall"])
        temperature = float(weather["temperature"])

        recommended_water = _clamp(
            (target - soil_before) * 0.48
            + max(0.0, temperature - 31.0) * 0.18
            - (rain_forecast / 100.0) * 3.2
            - rainfall * 0.35,
            (0.0, MAX_WATER_MM),
        )

        moisture_error = abs(soil_after - target)
        crop_health = _clamp(1.0 - moisture_error / 28.0, (0.0, 1.0))
        if target - 6.0 <= soil_after <= target + 6.0:
            crop_health = _clamp(crop_health + 0.12, (0.0, 1.0))

        water_efficiency = _clamp(1.0 - abs(applied_water - recommended_water) / MAX_WATER_MM, (0.0, 1.0))

        if rain_forecast >= 60.0 or rainfall >= 4.0:
            rain_awareness = _clamp(1.0 - applied_water / 8.0, (0.0, 1.0))
        else:
            rain_awareness = _clamp(0.45 + recommended_water / (MAX_WATER_MM * 1.8), (0.0, 1.0))

        overwatering_penalty = max(0.0, soil_after - (target + 8.0)) / 18.0
        overwatering_penalty += max(0.0, applied_water - recommended_water - 2.5) / 10.0

        underwatering_penalty = max(0.0, (target - 10.0) - soil_after) / 20.0
        if soil_before < target - 12.0 and applied_water <= 1.0:
            underwatering_penalty += 0.18

        extreme_action_penalty = 0.0
        if applied_water >= MAX_WATER_MM and recommended_water < 8.0:
            extreme_action_penalty += 0.12
        if applied_water == 0.0 and soil_before < target - 8.0 and rain_forecast < 65.0:
            extreme_action_penalty += 0.10

        budget_guard_penalty = 0.0
        if requested_water > applied_water and requested_water > self.water_budget + applied_water:
            budget_guard_penalty = 0.05

        reward = (
            0.55 * crop_health
            + 0.25 * water_efficiency
            + 0.20 * rain_awareness
            - overwatering_penalty
            - underwatering_penalty
            - extreme_action_penalty
            - budget_guard_penalty
        )
        reward = _clamp(reward, (-1.0, 1.0))

        reward_breakdown = {
            "crop_health": float(round(crop_health, 6)),
            "water_efficiency": float(round(water_efficiency, 6)),
            "rain_awareness": float(round(rain_awareness, 6)),
            "overwatering_penalty": float(round(overwatering_penalty, 6)),
            "underwatering_penalty": float(round(underwatering_penalty, 6)),
            "extreme_action_penalty": float(round(extreme_action_penalty, 6)),
            "budget_guard_penalty": float(round(budget_guard_penalty, 6)),
            "recommended_water_mm": float(round(recommended_water, 6)),
            "target_moisture": float(round(target, 6)),
            "zone_id": float(zone_id),
        }
        return float(round(reward, 6)), reward_breakdown

    def _build_explanation(
        self,
        zone_id: int,
        soil_before: float,
        soil_after: float,
        applied_water: float,
        target: float,
        weather: Mapping[str, float],
        reward_breakdown: Mapping[str, float],
        adjustments: List[str],
    ) -> Dict[str, Any]:
        soil_condition = (
            f"Zone {zone_id + 1} moved from {soil_before:.1f}% to {soil_after:.1f}% soil moisture "
            f"and is now {_soil_label(soil_after, target)} relative to the {target:.1f}% target."
        )
        weather_condition = (
            f"The weather was {_weather_label(float(weather['temperature']), float(weather['rain_forecast']), float(weather['rainfall']))}: "
            f"{float(weather['temperature']):.1f} C, forecast rain {float(weather['rain_forecast']):.0f}%, "
            f"observed rainfall {float(weather['rainfall']):.1f} mm."
        )

        reason_parts = [
            f"Applied {applied_water:.1f} mm to protect crop health around the {target:.1f}% target.",
        ]
        if float(weather["rain_forecast"]) >= 60.0:
            reason_parts.append("The plan stayed conservative because rain probability was elevated.")
        if float(weather["temperature"]) >= 33.0:
            reason_parts.append("Heat stress increased expected moisture loss.")
        if soil_before < target - 10.0:
            reason_parts.append("Dry soil required a recovery response.")
        elif soil_before > target + 8.0:
            reason_parts.append("Existing moisture levels called for restraint.")
        if adjustments:
            reason_parts.append("Safety validation adjusted the requested action before execution.")

        decision = f"Decision: irrigate Zone {zone_id + 1} with {applied_water:.1f} mm."
        reason = " ".join(reason_parts)
        summary = f"{soil_condition} {weather_condition} {decision} {reason}"

        return {
            "day": int(self.day + 1),
            "zone_id": int(zone_id),
            "soil_condition": soil_condition,
            "weather_condition": weather_condition,
            "decision_taken": decision,
            "reason_for_water_allocation": reason,
            "summary": summary,
            "reward_breakdown": dict(reward_breakdown),
            "action_adjustments": adjustments,
        }

    def _target_for_zone(self, zone_id: int) -> float:
        zone_bias = ((zone_id % 3) - 1) * 1.5
        return float(self.config.target_moisture + zone_bias)
