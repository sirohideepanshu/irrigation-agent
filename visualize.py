import matplotlib.pyplot as plt

def plot_rewards(rewards):
    steps = list(range(len(rewards)))

    plt.figure(figsize=(10, 5))

    # Main line
    plt.plot(steps, rewards, marker='o', linewidth=2)

    # Fill area under curve
    plt.fill_between(steps, rewards, alpha=0.2)

    # Labels
    plt.title("🌱 Irrigation Agent Performance Over Time", fontsize=14)
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    # Grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Avg line
    avg_reward = sum(rewards) / len(rewards)
    plt.axhline(avg_reward, linestyle='--', label=f"Avg: {avg_reward:.2f}")

    plt.legend()
    plt.tight_layout()

    plt.show()