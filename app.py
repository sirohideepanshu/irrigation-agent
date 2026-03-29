import gradio as gr
import matplotlib.pyplot as plt
import io
import sys

from inference import run_all


def run_simulation():
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        # ✅ IMPORTANT: capture returned values
        scores, overall, task_rewards = run_all()

    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"ERROR: {str(e)}", None

    sys.stdout = sys.__stdout__
    logs = buffer.getvalue()

    # -----------------------------
    # MULTI-TASK GRAPH
    # -----------------------------
    plt.figure()

    for task_name, rewards in task_rewards.items():
        plt.plot(rewards, label=task_name.upper())

    plt.title("Performance Across Tasks")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    plot_path = "output_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # -----------------------------
    # FINAL TEXT (summary)
    # -----------------------------
    summary = f"""

--------------------------------
FINAL SCORES

Easy   : {scores['easy']:.3f}
Medium : {scores['medium']:.3f}
Hard   : {scores['hard']:.3f}

Overall Score: {overall:.3f}
"""

    return logs + summary, plot_path


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(
    title="Smart Irrigation AI - Tensor Titans",
    css="""
    .gradio-container {
        background-image: url('https://images.unsplash.com/photo-1500382017468-9049fed747ef');
        background-size: cover;
        background-position: center;
    }

    .overlay {
        background: rgba(0,0,0,0.55);
        padding: 20px;
        border-radius: 12px;
    }
"""
) as demo:

    with gr.Column(elem_classes="overlay"):
        gr.Markdown("# Smart Irrigation AI - Tensor Titans")

        btn = gr.Button("Run Simulation")

        output_text = gr.Textbox(
            label="Simulation Logs",
            lines=20
        )

        output_plot = gr.Image(label="Performance Graph")

        btn.click(run_simulation, outputs=[output_text, output_plot])


demo.launch(server_name="127.0.0.1", server_port=7860)