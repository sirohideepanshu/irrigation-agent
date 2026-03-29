from inference import SmartIrrigationAgent


class Agent:
    def __init__(self):
        self.agent = SmartIrrigationAgent()

    def act(self, observation):
        actions = []

        soil_list = observation["soil_moisture"]

        for zone_id in range(len(soil_list)):
            action = self.agent.get_action(observation, zone_id)
            actions.append(action)

        return actions