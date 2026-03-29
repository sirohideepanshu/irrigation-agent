import numpy as np

class IrrigationEnv:
    def __init__(self):
        self.zones = 7
        self.max_steps = 40
        self.water_budget = 120

    def reset(self):
        self.steps = 0
        self.water_used = 0
        self.state = {
            "soil_moisture": np.random.randint(10, 80, self.zones).tolist(),
            "rain_forecast": np.random.choice([0, 1]),
            "temperature": np.random.randint(20, 40)
        }
        return self.state

    def step(self, action):
        zone = action["zone_id"]
        water = action["water_mm"]

        self.water_used += water

        evap_loss = (self.state["temperature"] - 20) * 0.1
        effective_water = max(0, water - evap_loss)

        if self.state["rain_forecast"]:
            effective_water *= 0.6

        self.state["soil_moisture"][zone] += effective_water

        moisture = self.state["soil_moisture"][zone]

        reward = 1 - abs(60 - moisture) / 60

        if self.water_used > self.water_budget:
            reward -= 0.5

        if moisture > 90:
            reward -= 0.4  # overwatering penalty

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, reward, done, {}

def create_env():
    return IrrigationEnv()