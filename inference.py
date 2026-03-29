import json

from tasks.easy import create_env as create_easy
from tasks.medium import create_env as create_medium
from tasks.hard import create_env as create_hard

from grader.grader import evaluate
from visualize import plot_rewards


class SmartIrrigationAgent:
    def __init__(self):
        self.prev_soil = {}

    def predict_soil(self, soil, water, rain, temp):
        water_effect = 0.8 * water
        rain_effect = 4 if rain == 1 else 0
        evaporation = max(0, (temp - 25) * 0.25)

        return soil + water_effect + rain_effect - evaporation

    def get_action(self, state, zone_id):
        soil = state["soil_moisture"][zone_id]
        prev = self.prev_soil.get(zone_id, soil)

        delta = soil - prev

        rain = int(state.get("rain_forecast", 0))
        temp = float(state.get("temperature", 25))

        TARGET = 52
        LOW = 46
        HIGH = 58

        # ---------------------------
        # BASE CONTROL (smooth)
        # ---------------------------
        error = TARGET - soil
        water = 0.5 * error

        # ---------------------------
        # LOW SOIL HANDLING (CONTROLLED)
        # ---------------------------
        if soil < 25:
            water = 8
        elif soil < 35:
            water += 3
        elif soil < 45:
            water += 1

        # ---------------------------
        # RAIN INTELLIGENCE
        # ---------------------------
        if rain == 1:
            water *= 0.7
            if soil > 52:
                water -= 1

        # ---------------------------
        # TEMPERATURE
        # ---------------------------
        if temp > 35:
            water += 2

        # ---------------------------
        # PREDICTION CORRECTION
        # ---------------------------
        predicted = self.predict_soil(soil, water, rain, temp)

        if predicted > HIGH:
            water -= 2
        elif predicted < LOW:
            water += 2

        # ---------------------------
        # OVERWATER CONTROL
        # ---------------------------
        if delta > 5:
            water -= 2

        # ---------------------------
        # HIGH SOIL
        # ---------------------------
        if soil > 65:
            water = 0

        # ---------------------------
        # SAFETY
        # ---------------------------
        if soil < 40 and water < 1:
            water = 1

        # ---------------------------
        # Clamp
        # ---------------------------
        water = max(0, min(12, int(round(water))))

        print(
            f"    Zone {zone_id} | Soil: {soil:.2f} | Delta: {delta:.2f} | "
            f"Rain: {rain} | Temp: {temp} | Pred: {predicted:.2f} | Water: {water}"
        )

        self.prev_soil[zone_id] = soil

        return {
            "zone_id": zone_id,
            "water_mm": water
        }


def run_task(task_name, create_env_fn):
    print(f"\nRunning {task_name.upper()} TASK")
    print("=" * 60)

    env = create_env_fn()
    state = env.reset()

    agent = SmartIrrigationAgent()

    total_reward = 0
    steps = 0
    rewards = []

    while True:
        for zone_id in range(len(state["soil_moisture"])):

            action = agent.get_action(state, zone_id)

            state, reward, done, info = env.step(action)

            extra_state = {
                k: int(v) if "int" in str(type(v)) else float(v) if "float" in str(type(v)) else v
                for k, v in state.items() if k != "soil_moisture"
            }

            print(
                f"[{task_name}] Step {steps:02d} | Zone {zone_id} | "
                f"Soil: {state['soil_moisture'][zone_id]:.2f} | "
                f"Water: {action['water_mm']} | Reward: {reward:.3f} | "
                f"Extra: {extra_state}"
            )

            total_reward += reward
            rewards.append(reward)
            steps += 1

            if done:
                break

        if done:
            break

    score = evaluate(total_reward, steps, state)

    print(f"\n{task_name.upper()} RESULTS")
    print("-" * 40)
    print(f"Score           : {score:.3f}")
    print(f"Avg Reward      : {sum(rewards)/len(rewards):.3f}")
    print(f"Max Reward      : {max(rewards):.3f}")
    print(f"Min Reward      : {min(rewards):.3f}")
    print("-" * 40)

    plot_rewards(rewards)

    return score, rewards, []


def run_all():
    scores = {}
    task_rewards = {}

    for name, fn in [
        ("easy", create_easy),
        ("medium", create_medium),
        ("hard", create_hard),
    ]:
        score, rewards, logs = run_task(name, fn)

        scores[name] = score
        task_rewards[name] = rewards

    overall = sum(scores.values()) / len(scores)

    return scores, overall, task_rewards


