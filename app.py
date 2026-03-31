from __future__ import annotations
import os
import sys
import types

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_DISABLE_AUDIO"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from functools import lru_cache

# Prevent optional audio imports from loading pydub/audioop on Spaces.
if "pydub" not in sys.modules:
    pydub_stub = types.ModuleType("pydub")
    pydub_stub.AudioSegment = None
    sys.modules["pydub"] = pydub_stub

import gradio as gr
import matplotlib

matplotlib.use("Agg")

from inference import run_all, stream_all
from visualize import (
    build_overview_reward_plot,
    build_overview_soil_plot,
    build_overview_water_plot,
    build_reward_plot,
    build_soil_plot,
    build_water_plot,
    create_empty_figure,
)


TASK_ORDER = ("easy", "medium", "hard")
BENCHMARK_SEEDS = (2026, 2027, 2028)


APP_CSS = """
body, .gradio-container {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-family: "IBM Plex Sans", ui-sans-serif, sans-serif;
    min-height: 100vh;
    position: relative;
}
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.78) 0%, rgba(2, 6, 23, 0.88) 45%, rgba(2, 6, 23, 0.94) 100%),
        linear-gradient(135deg, rgba(15, 23, 42, 0.42) 0%, rgba(30, 41, 59, 0.18) 48%, rgba(2, 6, 23, 0.58) 100%),
        url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=1800&q=80") center center / cover no-repeat;
    filter: blur(5px) saturate(0.78) brightness(0.52);
    transform: scale(1.04);
    z-index: -2;
}
body::after {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.01) 32%, transparent 55%),
        linear-gradient(180deg, rgba(2, 6, 23, 0.18) 0%, rgba(2, 6, 23, 0.08) 100%);
    z-index: -1;
    pointer-events: none;
}
.app-shell {
    max-width: 1380px;
    margin: 0 auto;
}
.hero {
    padding: 24px 8px 10px 8px;
}
.hero h1 {
    font-size: 2.2rem;
    margin-bottom: 8px;
}
.hero p {
    color: #94a3b8;
    margin: 0;
}
.panel {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.68) 0%, rgba(15, 23, 42, 0.58) 100%);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 18px 48px rgba(2, 6, 23, 0.34);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}
.muted {
    color: #94a3b8;
}
.gradio-container .block,
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-group,
.gradio-container .gr-form,
.gradio-container .gr-panel {
    background: transparent;
}
.gradio-container .gr-markdown,
.gradio-container label,
.gradio-container .gradio-textbox,
.gradio-container .gradio-dropdown,
.gradio-container .gradio-slider,
.gradio-container .gradio-button {
    color: #e2e8f0 !important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    background: rgba(15, 23, 42, 0.58) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    color: #e2e8f0 !important;
    backdrop-filter: blur(8px);
}
.gradio-container button {
    box-shadow: 0 10px 30px rgba(2, 6, 23, 0.28);
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
}
.gradio-container button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 34px rgba(34, 197, 94, 0.22);
    border-color: rgba(74, 222, 128, 0.42) !important;
}
.gradio-container [role="tab"],
.gradio-container [role="tablist"] {
    background: rgba(15, 23, 42, 0.42) !important;
    backdrop-filter: blur(10px);
}
.panel,
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    transition: box-shadow 180ms ease, border-color 180ms ease, transform 180ms ease;
}
.log-box textarea {
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
    background: rgba(15, 23, 42, 0.64) !important;
}
"""


def launch_with_fallback(demo_app: gr.Blocks) -> None:
    launch_kwargs = {
        "show_error": True,
    }

    server_name = os.getenv("GRADIO_SERVER_NAME")
    server_port = os.getenv("GRADIO_SERVER_PORT")

    if server_name:
        launch_kwargs["server_name"] = server_name
    if server_port:
        launch_kwargs["server_port"] = int(server_port)

    if not server_name:
        launch_kwargs["inbrowser"] = True

    demo_app.launch(**launch_kwargs)


def _empty_task_outputs():
    return tuple(
        create_empty_figure(f"{task.title()} {metric}", "Run simulation to generate insights")
        for task in TASK_ORDER
        for metric in ("Reward Trend", "Soil Moisture", "Water Usage")
    )


def _empty_overview_outputs():
    return (
        create_empty_figure("Reward Overview", "Run simulation to generate insights"),
        create_empty_figure("Soil Moisture Overview", "Run simulation to generate insights"),
        create_empty_figure("Water Usage Overview", "Run simulation to generate insights"),
    )


def _build_plot_outputs(summary):
    plots = []
    tasks = summary.get("tasks", {}) if summary else {}

    for task_name in TASK_ORDER:
        task_data = tasks.get(task_name, {})
        target = float(task_data.get("target_soil", 55.0))
        plots.extend(
            [
                build_reward_plot(task_name, task_data.get("rewards", [])),
                build_soil_plot(
                    task_name,
                    task_data.get("avg_soil", []),
                    task_data.get("focus_soil", []),
                    target,
                ),
                build_water_plot(
                    task_name,
                    task_data.get("water_usage", []),
                    task_data.get("cumulative_water", []),
                ),
            ]
        )

    return tuple(plots)


def _build_overview_outputs(summary):
    return (
        build_overview_reward_plot(summary),
        build_overview_soil_plot(summary),
        build_overview_water_plot(summary),
    )


@lru_cache(maxsize=1)
def _benchmark_markdown():
    seed_scores = []

    try:
        for seed in BENCHMARK_SEEDS:
            result = run_all({"seed": seed})
            if isinstance(result, tuple):
                if len(result) >= 2:
                    overall = result[1]
                else:
                    raise ValueError("Benchmark result tuple is missing the overall score.")
            else:
                overall = result.get("overall", 0.0)
            seed_scores.append((seed, float(overall)))
    except Exception as exc:
        return f"### 🏆 Performance Benchmark\nBenchmark unavailable: `{exc}`"

    if not seed_scores:
        return "### 🏆 Performance Benchmark\nBenchmark unavailable."

    best_seed, best_score = max(seed_scores, key=lambda item: item[1])
    average_score = sum(score for _, score in seed_scores) / len(seed_scores)
    lines = ["### 🏆 Performance Benchmark"]
    lines.extend(f"- Seed {seed}: `{score:.3f}`" for seed, score in seed_scores)
    lines.append(f"- Best: `Seed {best_seed} -> {best_score:.3f}`")
    lines.append(f"- Average Score: `{average_score:.3f}`")
    return "\n".join(lines)


def _score_markdown(summary):
    if not summary or not summary.get("scores"):
        return "### Final Scores\nRun the simulator to generate scores."

    scores = summary["scores"]
    overall = float(summary["overall"])
    if overall >= 0.80:
        badge = "🟢 Excellent"
    elif overall >= 0.70:
        badge = "🟡 Good"
    else:
        badge = "🔴 Needs Improvement"

    return (
        "### Final Scores\n"
        f"- Easy: `{scores.get('easy', 0.0):.3f}`\n"
        f"- Medium: `{scores.get('medium', 0.0):.3f}`\n"
        f"- Hard: `{scores.get('hard', 0.0):.3f}`\n"
        f"- Overall: `{overall:.3f}`\n\n"
        f"**Performance Badge:** {badge}"
    )


def _system_intelligence_markdown(summary):
    if not summary or not summary.get("tasks"):
        return (
            "### 🌱 System Intelligence Insights\n"
            "- Water Efficiency Score: `0.000`\n"
            "- Stability Score: `0.000`\n"
            "- Risk Level: `MEDIUM`"
        )

    tasks = summary.get("tasks", {})
    total_water = 0.0
    max_possible_water = 0.0
    rewards = []

    for task_data in tasks.values():
        task_water = float(task_data.get("total_water", 0.0))
        steps = int(task_data.get("steps", 0))
        total_water += task_water
        max_possible_water += float(steps * 12.0)
        rewards.extend(float(value) for value in task_data.get("rewards", []))

    water_efficiency = 0.0
    if max_possible_water > 0:
        water_efficiency = max(0.0, min(1.0, 1.0 - (total_water / max_possible_water)))

    stability_score = 0.0
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        stability_score = max(0.0, min(1.0, 1.0 / (1.0 + variance)))

    hard_score = float(summary.get("scores", {}).get("hard", 0.0))
    if hard_score < 0.5:
        risk_level = "HIGH"
    elif hard_score < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return (
        "### 🌱 System Intelligence Insights\n"
        f"- Water Efficiency Score: `{water_efficiency:.3f}`\n"
        f"- Stability Score: `{stability_score:.3f}`\n"
        f"- Risk Level: `{risk_level}`"
    )


def _status_markdown(event):
    if not event:
        return "### Live Status\nReady for a new simulation."

    if event["type"] == "complete":
        return "### Live Status\nSimulation complete. All tasks finished successfully."

    metrics = event["metrics"]
    return (
        "### Live Status\n"
        f"- Active task: `{event['task_name'].title()}`\n"
        f"- Completed steps: `{metrics['steps']}`\n"
        f"- Selected zone: `Zone {metrics['selected_zone'] + 1}`\n"
        f"- Running avg reward: `{metrics['avg_reward'] if metrics['steps'] else 0.0:.3f}`"
    )


def _warning_markdown(summary):
    if not summary:
        return "### Anomaly Detection\nNo overwatering, underwatering, or safety anomalies detected."

    warnings = []
    for task_name in TASK_ORDER:
        warnings.extend(summary.get("tasks", {}).get(task_name, {}).get("warnings", []))

    if not warnings:
        return "### Anomaly Detection\nNo overwatering, underwatering, or safety anomalies detected."

    latest = warnings[-6:]
    warning_lines = "\n".join(f"- ⚠️ {warning}" for warning in latest)
    return f"### Anomaly Detection\n{warning_lines}"


def _logs_text(summary):
    if not summary:
        return ""

    lines = []
    for task_name in TASK_ORDER:
        lines.extend(summary.get("tasks", {}).get(task_name, {}).get("logs", []))
    return "\n".join(lines)


def _current_summary_markdown(summary):
    if not summary or not summary.get("tasks"):
        return (
            "### Task Snapshot\n"
            "Task metrics will stream here while the simulation is running.\n\n"
            "Hard task represents high-uncertainty environments with aggressive weather variability."
        )

    lines = ["### Task Snapshot"]
    for task_name in TASK_ORDER:
        task_data = summary["tasks"].get(task_name)
        if not task_data:
            lines.append(f"- {task_name.title()}: waiting to start")
            continue

        lines.append(
            f"- {task_name.title()}: score `{task_data.get('score', 0.0):.3f}` | "
            f"avg reward `{task_data.get('avg_reward', 0.0):.3f}` | "
            f"water `{task_data.get('total_water', 0.0):.1f}` mm"
        )
    lines.append("")
    lines.append("Hard task represents high-uncertainty environments with aggressive weather variability.")
    return "\n".join(lines)


def _build_output_tuple(event, summary, update_plots=True):
    overview_plots = _build_overview_outputs(summary) if update_plots else tuple(gr.skip() for _ in range(3))
    task_plots = _build_plot_outputs(summary) if update_plots else tuple(gr.skip() for _ in range(9))
    return (
        _status_markdown(event),
        _score_markdown(summary),
        _current_summary_markdown(summary),
        _system_intelligence_markdown(summary),
        _warning_markdown(summary),
        _logs_text(summary),
        *overview_plots,
        *task_plots,
    )


def run_simulation(temperature, rain_forecast, soil_moisture, zone_label):
    selected_zone = int(str(zone_label).split()[-1]) - 1
    config = {
        "temperature": temperature,
        "rain_forecast": rain_forecast,
        "soil_moisture": soil_moisture,
        "selected_zone": selected_zone,
        "seed": 2026,
    }

    try:
        initial_event = None
        initial_summary = {"scores": {}, "overall": 0.0, "tasks": {}}
        yield _build_output_tuple(initial_event, initial_summary, update_plots=True)

        for event in stream_all(config):
            summary = event.get("summary", initial_summary)
            should_refresh_plots = (
                event["type"] in {"task_complete", "complete"}
                or (event["type"] == "step" and int(event["metrics"]["steps"]) % 1 == 0)
            )
            yield _build_output_tuple(event, summary, update_plots=should_refresh_plots)

    except Exception as exc:
        error_outputs = (
            "### Live Status\nSimulation failed.",
            "### Final Scores\nNo scores available because the simulation failed.",
            "### Task Snapshot\nThe run stopped before completion.",
            "### 🌱 System Intelligence Insights\nInsights unavailable because the run failed.",
            "### Anomaly Detection\nWarnings unavailable because the run failed.",
            f'{{"error": "{str(exc)}"}}',
            *_empty_overview_outputs(),
            *_empty_task_outputs(),
        )
        yield error_outputs


with gr.Blocks(title="Smart Irrigation AI", theme=gr.themes.Base(), css=APP_CSS) as demo:
    gr.HTML(
        """
        <div class="app-shell">
          <div class="hero">
            <h1>Smart Irrigation AI Control Center</h1>
            <p>AI-powered irrigation system that reduces water waste while maintaining optimal crop health under uncertain weather conditions.</p>
          </div>
        </div>
        """
    )

    with gr.Row(elem_classes="app-shell"):
        with gr.Column(scale=4):
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Simulation Controls")
                gr.Markdown(
                    "Tune the climate and selected zone conditions, then stream the AI controller across easy, medium, and hard scenarios.",
                    elem_classes="muted",
                )
                temperature = gr.Slider(-10, 60, value=30, step=1, label="Temperature (C)")
                rain_forecast = gr.Slider(0, 100, value=35, step=1, label="Rain Forecast (%)")
                soil_moisture = gr.Slider(0, 100, value=45, step=1, label="Selected Zone Soil Moisture (%)")
                zone_selector = gr.Dropdown(
                    choices=[f"Zone {index}" for index in range(1, 8)],
                    value="Zone 1",
                    label="Zone Selection",
                )
                with gr.Row():
                    run_button = gr.Button("Run Live Simulation", variant="primary")
                    smart_demo_button = gr.Button("🚀 Run Smart Demo", variant="secondary")

            with gr.Group(elem_classes="panel"):
                status_md = gr.Markdown("### Live Status\nReady for a new simulation.")
                score_md = gr.Markdown("### Final Scores\nRun the simulator to generate scores.")
                snapshot_md = gr.Markdown("### Task Snapshot\nTask metrics will stream here while the simulation is running.")
                insights_md = gr.Markdown("### 🌱 System Intelligence Insights\n- Water Efficiency Score: `0.000`\n- Stability Score: `0.000`\n- Risk Level: `MEDIUM`")
                benchmark_md = gr.Markdown(_benchmark_markdown())

            with gr.Group(elem_classes="panel"):
                warning_md = gr.Markdown("### Anomaly Detection\nNo overwatering, underwatering, or safety anomalies detected.")

        with gr.Column(scale=6):

            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Overview Graphs")
                overview_reward_plot = gr.Plot(label="Reward Overview")
                overview_soil_plot = gr.Plot(label="Soil Moisture Overview")
                overview_water_plot = gr.Plot(label="Water Usage Overview")

            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Structured Logs")
                logs_box = gr.Textbox(
                    label="Live JSON Logs",
                    lines=18,
                    max_lines=24,
                    elem_classes="log-box",
                )

            with gr.Tabs():
                reward_plots = {}
                soil_plots = {}
                water_plots = {}

                for task_name in TASK_ORDER:
                    with gr.Tab(f"{task_name.title()} Task"):
                        with gr.Row():
                            reward_plots[task_name] = gr.Plot(label=f"{task_name.title()} Reward Trend")
                            soil_plots[task_name] = gr.Plot(label=f"{task_name.title()} Soil Moisture")
                            water_plots[task_name] = gr.Plot(label=f"{task_name.title()} Water Usage")

    with gr.Row(elem_classes="app-shell"):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel"):
                gr.Markdown(
                    "### 🌍 Why This Matters\n"
                    "- Saves water in drought-prone regions\n"
                    "- Prevents crop damage from overwatering\n"
                    "- Adapts to unpredictable weather patterns\n"
                    "- Scalable to real farms using IoT sensors"
                )

    outputs = [
        status_md,
        score_md,
        snapshot_md,
        insights_md,
        warning_md,
        logs_box,
        overview_reward_plot,
        overview_soil_plot,
        overview_water_plot,
        reward_plots["easy"],
        soil_plots["easy"],
        water_plots["easy"],
        reward_plots["medium"],
        soil_plots["medium"],
        water_plots["medium"],
        reward_plots["hard"],
        soil_plots["hard"],
        water_plots["hard"],
    ]

    run_button.click(
        fn=run_simulation,
        inputs=[temperature, rain_forecast, soil_moisture, zone_selector],
        outputs=outputs,
    )

    smart_demo_button.click(
        fn=run_simulation,
        inputs=[
            gr.State(30),
            gr.State(40),
            gr.State(50),
            gr.State("Zone 2"),
        ],
        outputs=outputs,
    )


demo.queue()


if __name__ == "__main__":
    launch_with_fallback(demo)
