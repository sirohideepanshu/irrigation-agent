from __future__ import annotations

import json
from typing import Dict, Iterable, Mapping, Optional

from agent.policy import SmartIrrigationAgent, TASK_TARGETS
from grader.grader import evaluate
from tasks.easy import create_env as create_easy
from tasks.medium import create_env as create_medium
from tasks.hard import create_env as create_hard


# -----------------------------
# TASK SETUP
# -----------------------------
TASK_CREATORS = {
    "easy": create_easy,
    "medium": create_medium,
    "hard": create_hard,
}


# -----------------------------
# HELPERS
# -----------------------------
def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_config(config=None):
    config = dict(config or {})
    return {
        "temperature": float(_clamp(float(config.get("temperature", 30)), -10, 60)),
        "rain_forecast": float(_clamp(float(config.get("rain_forecast", 35)), 0, 100)),
        "soil_moisture": float(_clamp(float(config.get("soil_moisture", 45)), 0, 100)),
        "selected_zone": int(config.get("selected_zone", 0)),
        "seed": int(config.get("seed", 2026)),
    }


# -----------------------------
# RUN SINGLE TASK
# -----------------------------
def run_task(task_name, create_env_fn, config):
    env = create_env_fn(seed=config["seed"])
    state = env.reset(seed=config["seed"], initial_conditions=config)

    agent = SmartIrrigationAgent()

    total_reward = 0
    steps = 0

    rewards = []
    logs = []
    water_usage = []
    cumulative_water = []
    total_water = 0

    # ✅ IMPORTANT (fix soil graph)
    avg_soil = []
    focus_soil = []

    done = False

    while not done:
        for zone_id in range(len(state["soil_moisture"])):

            action = agent.get_action(
                state=state,
                zone_id=zone_id,
                task_name=task_name,
                remaining_budget=float(state.get("water_budget", 0)),
            )

            env_action = {
                "zone_id": zone_id,
                "water_mm": int(action["water_mm"]),
            }

            state, reward, done, info = env.step(env_action)

            total_reward += reward
            rewards.append(reward)
            steps += 1

            used = env.last_action.get("applied_water_mm", 0)
            total_water += used
            water_usage.append(used)
            cumulative_water.append(total_water)

            # ✅ FIX: Soil tracking
            avg_soil.append(sum(state["soil_moisture"]) / len(state["soil_moisture"]))
            focus_soil.append(state["soil_moisture"][config["selected_zone"]])

            logs.append(
                json.dumps({
                    "task": task_name,
                    "step": steps,
                    "zone": zone_id + 1,
                    "water": action["water_mm"],
                    "reward": round(reward, 4)
                })
            )

            if done:
                break

    score = evaluate(total_reward, steps, state)

    return {
        "score": score,
        "rewards": rewards,
        "logs": logs,
        "water_usage": water_usage,
        "cumulative_water": cumulative_water,
        "total_water": total_water,
        "avg_reward": total_reward / steps if steps else 0,
        "steps": steps,
        "target_soil": TASK_TARGETS.get(task_name, 55.0),
        "selected_zone": config.get("selected_zone", 0),

        # ✅ REQUIRED FOR UI
        "avg_soil": avg_soil,
        "focus_soil": focus_soil,

        # placeholders (UI safe)
        "warnings": [],
        "reasoning_logs": [],
    }


# -----------------------------
# RUN ALL TASKS
# -----------------------------
def run_all(config=None):
    config = normalize_config(config)

    scores = {}
    task_rewards = {}
    tasks_data = {}

    base_seed = config["seed"]

    for i, (task_name, create_env_fn) in enumerate(TASK_CREATORS.items()):
        task_config = dict(config)
        task_config["seed"] = base_seed + i * 17

        result = run_task(task_name, create_env_fn, task_config)

        scores[task_name] = result["score"]
        task_rewards[task_name] = result["rewards"]
        tasks_data[task_name] = result

    overall = sum(scores.values()) / len(scores)

    return scores, overall, task_rewards, tasks_data


# -----------------------------
# STREAM FOR UI
# -----------------------------
def stream_all(config=None):
    scores, overall, task_rewards, tasks_data = run_all(config)

    summary = {
        "scores": scores,
        "overall": overall,
        "tasks": tasks_data,
    }

    yield {
        "type": "complete",
        "summary": summary,
        "metrics": {},
    }


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    scores, overall, _, _ = run_all()

    print("\nFINAL SCORES")
    print("=" * 40)
    for k, v in scores.items():
        print(f"{k.upper()} : {v:.3f}")

    print(f"\nOVERALL : {overall:.3f}")