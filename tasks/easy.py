import numpy as np

class IrrigationEnv:
    def __init__(self):
        self.zones = 3
        self.max_steps = 20

    def reset(self):
        self.steps = 0
        self.state = {
            "soil_moisture": np.random.randint(30, 60, self.zones).tolist(),
            "rain_forecast": 0
        }
        return self.state

    def step(self, action):
        zone = action["zone_id"]
        water = action["water_mm"]

        self.state["soil_moisture"][zone] += water * 0.8

        reward = 1 - abs(50 - self.state["soil_moisture"][zone]) / 50

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, reward, done, {}

def create_env():
    return IrrigationEnv()