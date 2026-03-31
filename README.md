---
title: Smart Irrigation AI
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Smart Irrigation AI

Smart Irrigation AI is a reproducible irrigation-control environment built for OpenEnv-style evaluation. It simulates three difficulty tiers:

- `easy`: 3 zones with moderate weather variation and a shorter horizon
- `medium`: 5 zones with tighter water constraints and broader weather drift
- `hard`: 7 zones with hotter weather, larger uncertainty, and the strictest optimization pressure

The project includes:

- a validated irrigation environment in [`env/irrigation_env.py`](/Users/adityasingh/Desktop/irrigation-openenv/env/irrigation_env.py)
- a deterministic baseline runner in [`inference.py`](/Users/adityasingh/Desktop/irrigation-openenv/inference.py)
- a Gradio dashboard in [`app.py`](/Users/adityasingh/Desktop/irrigation-openenv/app.py)
- Docker support for Hugging Face Spaces

## Environment Description

Each step represents a control decision for one irrigation zone. The simulator updates:

- soil moisture per zone
- remaining water budget
- rain forecast and realized rainfall
- temperature drift and evaporation impact

The environment follows the expected contract:

- `reset()` returns the initial observation as a `dict`
- `step(action)` returns `(observation: dict, reward: float, done: bool, info: dict)`
- `state()` returns the current environment state as a `dict`

## Observation Space

Observation values are validated and clipped into safe bounds.

- `task_name`: current scenario name
- `soil_moisture`: list of zone moisture values in `%`, bounded to `0-100`
- `water_budget`: remaining water budget in `mm`
- `rain_forecast`: rain probability in `%`, bounded to `0-100`
- `temperature`: ambient temperature in `C`, bounded to `-10 to 60`
- `day`: current step index
- `max_steps`: episode horizon
- `target_moisture`: scenario target moisture
- `last_reward`: most recent reward
- `last_action`: most recent validated action payload

## Action Space

- `zone_id`: integer zone index for the irrigation target
- `water_mm`: requested water allocation in `mm`

Action safety rules:

- invalid zone ids are rejected
- negative water is clipped to `0`
- water is capped at `12 mm`
- water is clipped to the remaining budget when necessary

## Reward Function

The reward is intentionally non-constant and combines agronomic quality with operational discipline.

- `crop health`: rewards post-step soil moisture staying near the task target band
- `water efficiency`: rewards actions close to the recommended irrigation amount
- `rain awareness`: rewards conservative watering when rainfall is likely or already occurring
- `penalties`: reduce reward for overwatering, underwatering, extreme actions, and safety/budget violations

High-level formula:

```text
reward =
  0.55 * crop_health
  + 0.25 * water_efficiency
  + 0.20 * rain_awareness
  - overwatering_penalty
  - underwatering_penalty
  - extreme_action_penalty
  - budget_guard_penalty
```

## AI Reasoning Output

Each environment step returns a structured explanation in `info["explanation"]` with:

- current soil condition
- weather condition
- decision taken
- reason for water allocation
- reward breakdown

The Gradio UI surfaces the latest reasoning and streams structured JSON logs for all tasks.

## Baseline Results

Running the deterministic baseline with `python inference.py` currently prints:

```text
FINAL SCORES
Easy: 0.922
Medium: 0.736
Hard: 0.523
Overall: 0.727
```

The exact values are reproducible because the simulation uses fixed seeds by default.

## How To Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline:

```bash
python inference.py
```

Run the UI:

```bash
python app.py
```

## Docker / Hugging Face Spaces

The Docker image is compatible with Python 3.10 and uses the non-interactive `Agg` matplotlib backend.

Build locally:

```bash
docker build -t smart-irrigation-ai .
```

Run locally:

```bash
docker run -p 7860:7860 smart-irrigation-ai
```

For Hugging Face Docker Spaces:

- keep [`Dockerfile`](/Users/adityasingh/Desktop/irrigation-openenv/Dockerfile) at repo root
- keep [`requirements.txt`](/Users/adityasingh/Desktop/irrigation-openenv/requirements.txt) minimal
- use [`app.py`](/Users/adityasingh/Desktop/irrigation-openenv/app.py) as the entrypoint

## Project Files

- [`env/irrigation_env.py`](/Users/adityasingh/Desktop/irrigation-openenv/env/irrigation_env.py): shared validated irrigation environment
- [`tasks/easy.py`](/Users/adityasingh/Desktop/irrigation-openenv/tasks/easy.py): easy task wrapper
- [`tasks/medium.py`](/Users/adityasingh/Desktop/irrigation-openenv/tasks/medium.py): medium task wrapper
- [`tasks/hard.py`](/Users/adityasingh/Desktop/irrigation-openenv/tasks/hard.py): hard task wrapper
- [`inference.py`](/Users/adityasingh/Desktop/irrigation-openenv/inference.py): baseline inference and streaming metrics
- [`app.py`](/Users/adityasingh/Desktop/irrigation-openenv/app.py): Gradio dashboard
- [`visualize.py`](/Users/adityasingh/Desktop/irrigation-openenv/visualize.py): labeled reward, soil, and water plots
