from __future__ import annotations

from typing import Any, Mapping, Optional


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def evaluate(total_reward: float, steps: int, final_state: Mapping[str, Any]) -> float:
    if steps <= 0:
        return 0.0

    avg_reward = total_reward / steps
    soil_values = [float(value) for value in final_state.get("soil_moisture", [])]
    target = float(final_state.get("target_moisture", 55.0))
    remaining_budget = float(final_state.get("water_budget", 0.0))

    if soil_values:
        soil_balance = 1.0 - (sum(abs(value - target) for value in soil_values) / (len(soil_values) * 35.0))
    else:
        soil_balance = 0.0

    water_bonus = min(max(remaining_budget, 0.0) / 150.0, 0.12)
    score = 0.78 * _clamp(avg_reward) + 0.14 * _clamp(soil_balance) + water_bonus
    return _clamp(score)


def grade_all(summary: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    if summary is None:
        from inference import run_all_summary

        summary = run_all_summary()

    if "scores" not in summary:
        raise ValueError("grade_all expects a summary mapping with scores.")

    scores = {task: float(score) for task, score in dict(summary["scores"]).items()}
    overall = float(summary.get("overall", 0.0))
    return {
        "scores": scores,
        "overall": overall,
        "passed": overall >= 0.70,
    }
