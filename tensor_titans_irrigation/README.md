---
title: Tensor Titans Irrigation Environment Server
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Tensor Titans Irrigation Environment

This package exposes the smart irrigation environment as an OpenEnv-compatible FastAPI server.

## What The Server Provides

- deterministic resets for reproducible evaluation
- validated `zone_id` and `water_mm` actions
- irrigation observations with soil moisture, water budget, rain forecast, and temperature
- structured step metadata with reasoning and reward breakdowns

## Action Schema

- `zone_id`: irrigation zone index
- `water_mm`: requested irrigation amount in `mm` between `0` and `12`

## Observation Schema

- `soil_moisture`: moisture values for all zones
- `water_budget`: remaining water budget
- `rain_forecast`: forecasted rain probability
- `temperature`: current temperature
- `day`: current episode step
- `target_moisture`: control target for the active scenario
- `last_reasoning`: text summary of the latest decision

## Run Locally

```bash
docker build -t tensor-titans-irrigation-env:latest -f server/Dockerfile .
docker run -p 8000:8000 tensor-titans-irrigation-env:latest
```

## Deploy

From the package directory containing [`openenv.yaml`](/Users/adityasingh/Desktop/irrigation-openenv/tensor_titans_irrigation/openenv.yaml):

```bash
openenv push
```
