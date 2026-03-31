from __future__ import annotations

import matplotlib.pyplot as plt


TASK_COLORS = {
    "easy": "#4ade80",
    "medium": "#38bdf8",
    "hard": "#f97316",
}


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, color="#f8fafc", pad=12)
    ax.set_xlabel("Simulation Step", color="#cbd5e1")
    ax.set_ylabel(ylabel, color="#cbd5e1")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.18)
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")


def create_empty_figure(title: str, message: str):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#111827")
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#94a3b8", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color="#f8fafc", pad=12)
    fig.tight_layout()
    return fig


def build_reward_plot(task_name: str, rewards):
    if not rewards:
        return create_empty_figure(f"{task_name.title()} Reward Trend", "Run a simulation to populate reward data.")

    steps = list(range(1, len(rewards) + 1))
    color = TASK_COLORS.get(task_name, "#38bdf8")
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")

    ax.plot(steps, rewards, color=color, linewidth=2.4, marker="o", markersize=3.5)
    ax.fill_between(steps, rewards, color=color, alpha=0.18)
    ax.axhline(sum(rewards) / len(rewards), linestyle="--", linewidth=1.1, color="#f8fafc", alpha=0.65)

    _style_axes(ax, f"{task_name.title()} Reward Trend", "Reward")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def build_soil_plot(task_name: str, avg_soil, focus_soil, target: float):
    if not avg_soil:
        return create_empty_figure(f"{task_name.title()} Soil Moisture", "Run a simulation to populate soil data.")

    steps = list(range(1, len(avg_soil) + 1))
    color = TASK_COLORS.get(task_name, "#38bdf8")
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")

    ax.plot(steps, avg_soil, color=color, linewidth=2.4, label="Average soil")
    ax.plot(steps, focus_soil, color="#f8fafc", linewidth=1.8, linestyle="--", label="Selected zone")
    ax.axhspan(target - 6, target + 6, color="#22c55e", alpha=0.12, label="Target band")

    _style_axes(ax, f"{task_name.title()} Soil Moisture", "Soil Moisture (%)")
    ax.set_ylim(0, 100)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    return fig


def build_water_plot(task_name: str, water_usage, cumulative_water):
    if not water_usage:
        return create_empty_figure(f"{task_name.title()} Water Usage", "Run a simulation to populate water data.")

    steps = list(range(1, len(water_usage) + 1))
    color = TASK_COLORS.get(task_name, "#38bdf8")
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")

    ax.bar(steps, water_usage, color=color, alpha=0.55, label="Per-step water")
    ax.plot(steps, cumulative_water, color="#f8fafc", linewidth=2.0, label="Cumulative water")

    _style_axes(ax, f"{task_name.title()} Water Usage", "Water (mm)")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    return fig


def build_overview_reward_plot(summary):
    tasks = summary.get("tasks", {}) if summary else {}
    if not tasks:
        return create_empty_figure("Reward Overview", "Run a simulation to compare reward trends across tasks.")

    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#0f172a")
    plotted = False
    for task_name, task_data in tasks.items():
        rewards = task_data.get("rewards", [])
        if not rewards:
            continue
        plotted = True
        ax.plot(
            list(range(1, len(rewards) + 1)),
            rewards,
            linewidth=2.2,
            label=task_name.title(),
            color=TASK_COLORS.get(task_name, "#38bdf8"),
        )

    if not plotted:
        return create_empty_figure("Reward Overview", "Run a simulation to compare reward trends across tasks.")

    _style_axes(ax, "Reward Overview", "Reward")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    return fig


def build_overview_soil_plot(summary):
    tasks = summary.get("tasks", {}) if summary else {}
    if not tasks:
        return create_empty_figure("Soil Moisture Overview", "Run a simulation to compare soil moisture across tasks.")

    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#0f172a")
    plotted = False
    for task_name, task_data in tasks.items():
        avg_soil = task_data.get("avg_soil", [])
        if not avg_soil:
            continue
        plotted = True
        ax.plot(
            list(range(1, len(avg_soil) + 1)),
            avg_soil,
            linewidth=2.2,
            label=f"{task_name.title()} avg",
            color=TASK_COLORS.get(task_name, "#38bdf8"),
        )

    if not plotted:
        return create_empty_figure("Soil Moisture Overview", "Run a simulation to compare soil moisture across tasks.")

    _style_axes(ax, "Soil Moisture Overview", "Soil Moisture (%)")
    ax.set_ylim(0, 100)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    return fig


def build_overview_water_plot(summary):
    tasks = summary.get("tasks", {}) if summary else {}
    if not tasks:
        return create_empty_figure("Water Usage Overview", "Run a simulation to compare water use across tasks.")

    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#0f172a")
    plotted = False
    for task_name, task_data in tasks.items():
        cumulative_water = task_data.get("cumulative_water", [])
        if not cumulative_water:
            continue
        plotted = True
        ax.plot(
            list(range(1, len(cumulative_water) + 1)),
            cumulative_water,
            linewidth=2.2,
            label=task_name.title(),
            color=TASK_COLORS.get(task_name, "#38bdf8"),
        )

    if not plotted:
        return create_empty_figure("Water Usage Overview", "Run a simulation to compare water use across tasks.")

    _style_axes(ax, "Water Usage Overview", "Cumulative Water (mm)")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    return fig
