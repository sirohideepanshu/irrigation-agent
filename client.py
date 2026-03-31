from __future__ import annotations

from agent.policy import SmartIrrigationAgent


class Agent:
    def __init__(self):
        self.agent = SmartIrrigationAgent()

    def act(self, observation):
        actions = []
        soil_list = observation["soil_moisture"]
        remaining_budget = float(observation.get("water_budget", 0.0))
        task_name = str(observation.get("task_name", "medium"))

        for zone_id in range(len(soil_list)):
            action = self.agent.get_action(
                observation,
                zone_id,
                task_name=task_name,
                remaining_budget=remaining_budget,
            )
            actions.append(action)

        return actions
