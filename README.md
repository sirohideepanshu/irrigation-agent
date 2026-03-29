---
title: irrigation-agent
colorFrom: green
colorTo: blue
sdk: docker
---

## Problem Description
This project simulates a smart irrigation system where an AI agent manages water distribution across multiple zones based on soil moisture, weather, and crop health.

## Observation Space
- soil_moisture: List of moisture values per zone
- water_budget: remaining water

## Action Space

- zone_id: zone to irrigate
- water_mm: amount of water (0–12)

## Reward Function

 Reward Function

The reward function is designed to simulate real-world irrigation decision-making by balancing crop health, water efficiency, and environmental awareness.

 Crop Health (Primary Objective)

The main goal is to maintain optimal soil moisture for healthy crop growth.
The crop_health() function returns higher values when moisture is within the ideal range.

 This contributes 60% of the total reward.

  Water Efficiency

The system encourages using an optimal amount of water (~8 mm).
Using too little or too much reduces efficiency.

formula:
efficiency = 1 - |water - 8| / 8

 Prevents waste and promotes smart irrigation.


Rain Awareness (Environmental Intelligence)

The agent gets a bonus when:
	•	rainfall is high
	•	and it reduces irrigation accordingly

Encourages real-world adaptive behavior


Penalties (Critical Constraints)

Penalties are applied for:
	•	Overwatering during heavy rain
	•	Violating water budget

Ensures:
	•	resource conservation
	•	realistic constraints


Final Reward Formula:
reward = 0.6 * crop_health + 0.3 * efficiency + rain_bonus - penalties

## Tasks
Easy: single zone
Medium: multiple zones
Hard: constrained + optimization


##  Baseline Results
after running once:

Easy Task   : 0.928  
Medium Task : 0.804  
Hard Task   : 0.834  

Overall Score: 0.855

## Run
```bash
python inference.py