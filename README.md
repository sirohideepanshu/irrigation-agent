---
title: CropPulse AI
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
tags:
  - openenv
app_port: 8000
pinned: false
short_description: CropPulse AI — Where every drop listens to your crops.
---

# 🌱 CropPulse AI

*Where every drop listens to your crops.*

## 🌍 Why This Problem Matters

Irrigation decisions directly affect crop health, water consumption, and farm operating cost.

In real farms, farmers need to decide:
- which zone needs water first
- how much water should be applied
- whether incoming rainfall should reduce irrigation
- how to avoid both crop stress and water waste

CropPulse AI turns irrigation management into a premium, production-ready intelligence layer for modern farming. It models real-world water allocation decisions as an OpenEnv environment, making it useful for both agent evaluation and sustainable agriculture experimentation.

## 💡 What This Project Does

This project provides a **deterministic, multi-task CropPulse AI irrigation environment** with:
- `easy`, `medium`, and `hard` tasks
- reward-based irrigation simulation
- typed action and observation models
- deterministic programmatic grading
- a baseline inference script with reproducible scores
- Docker deployment for Hugging Face Spaces

The deployed submission is an **OpenEnv-compatible FastAPI server** exposed from [`server_app.py`](/Users/adityasingh/Desktop/CropPulse%20AI/server_app.py).

An optional local **Gradio dashboard** still exists in [`app.py`](/Users/adityasingh/Desktop/CropPulse%20AI/app.py) for demos, but the submitted Space is the environment server required by the hackathon.

## ✅ OpenEnv Interface

The core environment is implemented in [`irrigation_env.py`](/Users/adityasingh/Desktop/CropPulse%20AI/irrigation_env.py).

It follows the required interface:
- `reset()` returns the initial observation as a `dict`
- `step(action)` returns `(observation: dict, reward: float, done: bool, info: dict)`
- `state()` returns the current environment state

The root submission metadata is defined in [`openenv.yaml`](/Users/adityasingh/Desktop/CropPulse%20AI/openenv.yaml).

## 🧠 Tasks

The environment includes 3 graded tasks:

- `easy`: fewer zones, milder weather variation, simpler water balancing
- `medium`: more zones, tighter water constraints, broader weather drift
- `hard`: highest uncertainty, stronger temperature pressure, and stricter water allocation tradeoffs

These tasks are implemented in:
- [`tasks/easy.py`](/Users/adityasingh/Desktop/CropPulse%20AI/tasks/easy.py)
- [`tasks/medium.py`](/Users/adityasingh/Desktop/CropPulse%20AI/tasks/medium.py)
- [`tasks/hard.py`](/Users/adityasingh/Desktop/CropPulse%20AI/tasks/hard.py)

## 📦 Observation Space

Observations include:
- `soil_moisture`: moisture values for each irrigation zone
- `water_budget`: remaining available irrigation water
- `rain_forecast`: forecast probability for rainfall
- `temperature`: ambient temperature
- `day`: current step index
- `max_steps`: episode horizon
- `target_moisture`: target moisture band
- `last_reward`: reward from the previous step
- `last_action`: validated action applied by the environment

## 🎯 Action Space

Actions use:
- `zone_id`: integer zone index
- `water_mm`: irrigation amount in millimeters

Validation rules:
- invalid zones raise an error
- negative water requests are clipped to `0`
- irrigation is capped at `12 mm`
- actions cannot exceed remaining budget

Typed models are defined in:
- [`models.py`](/Users/adityasingh/Desktop/CropPulse%20AI/models.py)
- [`tensor_titans_irrigation/models.py`](/Users/adityasingh/Desktop/CropPulse%20AI/tensor_titans_irrigation/models.py)

## 🏆 Reward Function

The reward is designed to be meaningful across the full trajectory, not only at the episode end.

It combines:
- `crop health`: rewards moisture near the task target band
- `water efficiency`: discourages unnecessary irrigation
- `rain awareness`: reduces reward for watering aggressively before likely rainfall
- `penalties`: punishes overwatering, underwatering, budget misuse, and extreme actions

This makes the environment suitable for agent learning and for hackathon evaluation focused on realistic partial progress signals.

## 📊 Baseline Results

Running [`inference.py`](/Users/adityasingh/Desktop/CropPulse%20AI/inference.py) prints deterministic baseline scores:

```text
FINAL SCORES
Easy: 0.922
Medium: 0.736
Hard: 0.523
Overall: 0.727
```

## 🧪 Grading

Programmatic grading lives in [`grader/grader.py`](/Users/adityasingh/Desktop/CropPulse%20AI/grader/grader.py).

The grader:
- evaluates all 3 tasks
- produces scores in the `0.0` to `1.0` range
- returns an overall score
- is deterministic for the same seed/configuration

## 🚀 How To Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the baseline inference script

```bash
python inference.py
```

### 3. Run the OpenEnv server locally

```bash
uvicorn server_app:app --host 0.0.0.0 --port 8000
```

### 4. Optional: run the local Gradio dashboard

If you want the presentation dashboard for demos, run:

```bash
python app.py
```

## 🐳 Docker / Hugging Face Spaces

The root [`Dockerfile`](/Users/adityasingh/Desktop/CropPulse%20AI/Dockerfile) launches the OpenEnv-compatible FastAPI server on port `8000`.

Build locally:

```bash
docker build -t croppulse-ai .
```

Run locally:

```bash
docker run -p 8000:8000 croppulse-ai
```

For Hugging Face Spaces:
- the repository is configured as `sdk: docker`
- the app port is `8000`
- the Space should expose the OpenEnv HTTP server required by the hackathon validator

## 🖥️ Optional Demo UI

The included dashboard:
- visualizes reward trends
- shows soil moisture plots
- shows water usage plots
- helps explain task performance for demos and presentations

The dashboard code is in [`app.py`](/Users/adityasingh/Desktop/CropPulse%20AI/app.py).

## 🌱 Real-World Impact

This project demonstrates how AI can support:
- water conservation in drought-prone regions
- improved crop stability
- better response to weather uncertainty
- zone-specific irrigation planning
- scalable decision support for future IoT-enabled farms

## 🔮 Future Work

- live weather API integration
- crop-specific agronomic models
- IoT sensor ingestion
- reinforcement learning policy training
- multi-farm seasonal optimization

## 👥 Team

- Deepanshu Sirohi
- Sahas Rastohi
- Yashraj Gulyani
