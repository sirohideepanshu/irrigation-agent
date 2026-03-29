import numpy as np

class IrrigationEnv:
    def __init__(self):
        self.zones = 5
        self.max_steps = 30
        self.water_budget = 100

    def reset(self):
        self.steps = 0
        self.water_used = 0
        self.state = {
            "soil_moisture": np.random.randint(20, 70, self.zones).tolist(),
            "rain_forecast": np.random.choice([0, 1])
        }
        return self.state

    def step(self, action):
        zone = action["zone_id"]
        water = action["water_mm"]

        self.water_used += water

        if self.state["rain_forecast"]:
            water *= 0.5

        self.state["soil_moisture"][zone] += water * 0.7

        moisture = self.state["soil_moisture"][zone]

        reward = 1 - abs(55 - moisture) / 55

        if self.water_used > self.water_budget:
            reward -= 0.3

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, reward, done, {}

def create_env():
    return IrrigationEnv()