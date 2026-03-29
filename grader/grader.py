def evaluate(total_reward, steps, final_state):
    """
    Generic evaluation function for all tasks
    Returns score between 0 and 1
    """

    # Avoid division by zero
    if steps == 0:
        return 0

    # Base performance
    avg_reward = total_reward / steps

    # Water efficiency bonus
    water_bonus = 0

    if "water_budget" in final_state:
        remaining = max(0, final_state["water_budget"])
        water_bonus = min(remaining / 100, 0.2)

    # Final score
    score = avg_reward + water_bonus

    # Clamp
    score = max(0, min(score, 1))

    return score