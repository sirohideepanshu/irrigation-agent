from __future__ import annotations

from typing import Dict, List, Optional


TASK_TARGETS = {
    "easy": 52.0,
    "medium": 55.0,
    "hard": 58.0,
}


class SmartIrrigationAgent:
    def __init__(self) -> None:
        self.prev_soil: Dict[int, float] = {}

    def predict_soil(self, soil: float, water: float, rain_forecast: float, temp: float) -> float:
        water_effect = 0.82 * water
        rain_effect = (rain_forecast / 100.0) * 2.2
        evaporation = max(0.0, temp - 24.0) * 0.18
        crop_use = 1.1
        return soil + water_effect + rain_effect - evaporation - crop_use

    def detect_anomalies(
        self,
        soil: float,
        predicted_soil: float,
        target: float,
        rain_forecast: float,
        water: float,
    ) -> List[str]:
        warnings: List[str] = []

        if predicted_soil > target + 10.0 or (soil > target + 8.0 and water > 0.0):
            warnings.append("Overwatering risk")
        if predicted_soil < target - 10.0 or (soil < target - 12.0 and water <= 1.0):
            warnings.append("Underwatering risk")
        if rain_forecast >= 70.0 and water >= 5.0:
            warnings.append("Rain-adjustment warning")
        if water >= 11.0:
            warnings.append("Extreme action warning")

        return warnings

    def build_reasoning(
        self,
        zone_id: int,
        soil: float,
        target: float,
        rain_forecast: float,
        temp: float,
        trend: float,
        predicted_soil: float,
        water: int,
        warnings: List[str],
    ) -> Dict[str, object]:
        soil_condition = (
            f"Zone {zone_id + 1} is at {soil:.1f}% moisture against a {target:.1f}% target."
        )
        weather_condition = (
            f"Temperature is {temp:.1f} C with a {rain_forecast:.0f}% rain forecast."
        )
        decision_taken = f"Allocate {water} mm of water to Zone {zone_id + 1}."

        reasons: List[str] = []
        if soil < target - 10.0:
            reasons.append("Soil is significantly below target and needs recovery irrigation.")
        elif soil < target - 4.0:
            reasons.append("Soil is a bit dry, so a moderate correction is appropriate.")
        elif soil > target + 6.0:
            reasons.append("Soil is already moist, so irrigation stays conservative.")
        else:
            reasons.append("Soil is near target, so only a fine adjustment is needed.")

        if temp >= 33.0:
            reasons.append("High temperature increases evaporation risk.")
        if rain_forecast >= 60.0:
            reasons.append("Rain probability reduces the watering plan.")
        elif rain_forecast <= 20.0:
            reasons.append("Low rain probability supports proactive irrigation.")

        if trend <= -3.0:
            reasons.append("Recent moisture decline added a recovery bump.")
        elif trend >= 3.0:
            reasons.append("Recent moisture gains justified a smaller action.")

        reasons.append(f"Predicted post-step soil moisture is {predicted_soil:.1f}%.")
        if warnings:
            reasons.append(f"Warnings: {', '.join(warnings)}.")

        reason_for_water_allocation = " ".join(reasons)
        summary = f"{soil_condition} {weather_condition} {decision_taken} {reason_for_water_allocation}"

        return {
            "soil_condition": soil_condition,
            "weather_condition": weather_condition,
            "decision_taken": decision_taken,
            "reason_for_water_allocation": reason_for_water_allocation,
            "summary": summary,
        }

    def get_action(
        self,
        state: Dict[str, object],
        zone_id: int,
        task_name: str = "medium",
        remaining_budget: Optional[float] = None,
    ) -> Dict[str, object]:
        soil = float(state["soil_moisture"][zone_id])
        prev_soil = self.prev_soil.get(zone_id, soil)
        trend = soil - prev_soil

        rain_forecast = float(state.get("rain_forecast", 0.0))
        temp = float(state.get("temperature", 25.0))
        current_day = int(state.get("day", 0))
        max_steps = int(state.get("max_steps", 1))
        remaining_decisions = max(1, max_steps - current_day)

        task_target = TASK_TARGETS.get(task_name, 55.0)
        zone_bias = ((zone_id % 3) - 1) * 1.5
        target = task_target + zone_bias
        moisture_gap = target - soil

        water = max(0.0, moisture_gap * 0.52)
        water += max(0.0, temp - 31.0) * 0.20
        water -= (rain_forecast / 100.0) * 4.0

        if trend <= -3.0:
            water += 1.8
        elif trend >= 3.0:
            water -= 1.4

        if soil < target - 14.0:
            water += 2.5
        elif soil > target + 7.0:
            water = 0.0

        if rain_forecast >= 75.0 and soil >= target - 5.0:
            water = 0.0

        if remaining_budget is not None:
            sustainable_cap = remaining_budget / remaining_decisions
            if soil < target - 15.0:
                sustainable_cap += 1.5
            elif soil > target + 5.0:
                sustainable_cap -= 1.0

            sustainable_cap = max(0.0, min(12.0, sustainable_cap))
            water = min(water, sustainable_cap)

            if remaining_budget < 12.0:
                water = min(water, max(0.0, remaining_budget / 2.0))
            water = min(water, max(0.0, remaining_budget))

        water = max(0.0, min(12.0, water))
        predicted_soil = self.predict_soil(soil, water, rain_forecast, temp)

        if predicted_soil > target + 7.0:
            water = max(0.0, water - min(3.0, predicted_soil - (target + 7.0)))
            predicted_soil = self.predict_soil(soil, water, rain_forecast, temp)
        elif predicted_soil < target - 9.0 and remaining_budget not in {0.0, 0}:
            water = min(12.0, water + min(2.5, (target - 9.0) - predicted_soil))
            predicted_soil = self.predict_soil(soil, water, rain_forecast, temp)

        water = max(0, min(12, int(round(water))))
        predicted_soil = self.predict_soil(soil, float(water), rain_forecast, temp)

        warnings = self.detect_anomalies(soil, predicted_soil, target, rain_forecast, float(water))
        reasoning = self.build_reasoning(
            zone_id=zone_id,
            soil=soil,
            target=target,
            rain_forecast=rain_forecast,
            temp=temp,
            trend=trend,
            predicted_soil=predicted_soil,
            water=water,
            warnings=warnings,
        )

        self.prev_soil[zone_id] = soil

        return {
            "zone_id": zone_id,
            "water_mm": water,
            "predicted_soil": predicted_soil,
            "target_soil": target,
            "reasoning": reasoning["summary"],
            "reasoning_structured": reasoning,
            "warnings": warnings,
            "trend": trend,
            "temperature": temp,
            "rain_forecast": rain_forecast,
        }
