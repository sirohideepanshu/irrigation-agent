from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple

from agent.policy import SmartIrrigationAgent, TASK_TARGETS
from grader.grader import evaluate
from tasks.easy import create_env as create_easy
from tasks.medium import create_env as create_medium
from tasks.hard import create_env as create_hard


TASK_CREATORS = {
    "easy": create_easy,
    "medium": create_medium,
    "hard": create_hard,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = dict(config or {})
    return {
        "temperature": float(_clamp(float(config.get("temperature", 30)), -10, 60)),
        "rain_forecast": float(_clamp(float(config.get("rain_forecast", 35)), 0, 100)),
        "soil_moisture": float(_clamp(float(config.get("soil_moisture", 45)), 0, 100)),
        "selected_zone": int(config.get("selected_zone", 0)),
        "seed": int(config.get("seed", 2026)),
    }


def _env_runtime_metadata() -> Dict[str, Any]:
    return {
        "api_base_url": os.getenv("API_BASE_URL", ""),
        "model_name": os.getenv("MODEL_NAME", ""),
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
    }


def _maybe_init_openai_client() -> Any:
    api_base_url = os.getenv("API_BASE_URL")
    hf_token = os.getenv("HF_TOKEN")
    if not api_base_url or not hf_token:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    return OpenAI(base_url=api_base_url, api_key=hf_token)


def _emit_log(tag: str, payload: Mapping[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(dict(payload), sort_keys=False)}")


def run_task(task_name: str, create_env_fn, config: Mapping[str, Any]) -> Dict[str, Any]:
    env = create_env_fn(seed=config["seed"])
    state = env.reset(seed=config["seed"], initial_conditions=config)

    agent = SmartIrrigationAgent()

    total_reward = 0.0
    steps = 0

    rewards = []
    logs = []
    water_usage = []
    cumulative_water = []
    total_water = 0.0
    avg_soil = []
    focus_soil = []
    warnings = []
    reasoning_logs = []

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
            rewards.append(float(reward))
            steps += 1

            used = float(env.last_action.get("applied_water_mm", 0.0))
            total_water += used
            water_usage.append(used)
            cumulative_water.append(total_water)

            avg_soil.append(sum(state["soil_moisture"]) / len(state["soil_moisture"]))
            focus_index = int(config["selected_zone"]) % len(state["soil_moisture"])
            focus_soil.append(state["soil_moisture"][focus_index])

            warnings.extend(str(item) for item in info.get("action_validation", []))

            explanation = dict(info.get("explanation", {}))
            if explanation:
                reasoning_logs.append(explanation)

            log_record = {
                "task": task_name,
                "step": steps,
                "zone": zone_id + 1,
                "water": int(action["water_mm"]),
                "reward": round(float(reward), 4),
            }
            logs.append(json.dumps(log_record))

            _emit_log(
                "STEP",
                {
                    "task": task_name,
                    "step": steps,
                    "zone": zone_id + 1,
                    "water_mm": int(action["water_mm"]),
                    "reward": round(float(reward), 4),
                    "done": bool(done),
                },
            )

            if done:
                break

    score = evaluate(total_reward, steps, state)

    return {
        "score": float(score),
        "rewards": rewards,
        "logs": logs,
        "water_usage": water_usage,
        "cumulative_water": cumulative_water,
        "total_water": float(total_water),
        "avg_reward": float(total_reward / steps) if steps else 0.0,
        "steps": int(steps),
        "target_soil": float(TASK_TARGETS.get(task_name, 55.0)),
        "selected_zone": int(config.get("selected_zone", 0)),
        "avg_soil": avg_soil,
        "focus_soil": focus_soil,
        "warnings": warnings[-8:],
        "reasoning_logs": reasoning_logs,
    }


def run_all(config: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, float], float, Dict[str, list], Dict[str, Dict[str, Any]]]:
    config = normalize_config(config)

    scores: Dict[str, float] = {}
    task_rewards: Dict[str, list] = {}
    tasks_data: Dict[str, Dict[str, Any]] = {}

    base_seed = int(config["seed"])
    runtime_meta = _env_runtime_metadata()

    _emit_log(
        "START",
        {
            "seed": base_seed,
            "selected_zone": int(config["selected_zone"]),
            "temperature": float(config["temperature"]),
            "rain_forecast": float(config["rain_forecast"]),
            "soil_moisture": float(config["soil_moisture"]),
            "runtime": runtime_meta,
            "openai_client_initialized": bool(_maybe_init_openai_client()),
            "baseline_mode": "deterministic_heuristic",
        },
    )

    for i, (task_name, create_env_fn) in enumerate(TASK_CREATORS.items()):
        task_config = dict(config)
        task_config["seed"] = base_seed + i * 17

        result = run_task(task_name, create_env_fn, task_config)

        scores[task_name] = float(result["score"])
        task_rewards[task_name] = list(result["rewards"])
        tasks_data[task_name] = result

    overall = float(sum(scores.values()) / len(scores))
    _emit_log(
        "END",
        {
            "scores": {task: round(score, 3) for task, score in scores.items()},
            "overall": round(overall, 3),
        },
    )

    return scores, overall, task_rewards, tasks_data


def run_all_summary(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    scores, overall, task_rewards, tasks_data = run_all(config)
    return {
        "scores": scores,
        "overall": overall,
        "task_rewards": task_rewards,
        "tasks": tasks_data,
    }


def stream_all(config: Optional[Mapping[str, Any]] = None):
    summary = run_all_summary(config)
    yield {
        "type": "complete",
        "summary": {
            "scores": summary["scores"],
            "overall": summary["overall"],
            "tasks": summary["tasks"],
        },
        "metrics": {},
    }


def main(config: Optional[Mapping[str, Any]] = None):
    scores, overall, task_rewards, _ = run_all(config)

    print("\nFINAL SCORES")
    print(f"Easy: {scores['easy']:.3f}")
    print(f"Medium: {scores['medium']:.3f}")
    print(f"Hard: {scores['hard']:.3f}")
    print(f"Overall: {overall:.3f}")

    return scores, overall, task_rewards


if __name__ == "__main__":
    main()
