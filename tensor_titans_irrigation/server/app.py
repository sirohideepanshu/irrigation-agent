# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the CropPulse AI irrigation environment.

This module creates an HTTP server that exposes the TensorTitansIrrigationEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from fastapi.responses import HTMLResponse
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import TensorTitansIrrigationAction, TensorTitansIrrigationObservation
    from .tensor_titans_irrigation_environment import TensorTitansIrrigationEnvironment
except ModuleNotFoundError:
    from models import TensorTitansIrrigationAction, TensorTitansIrrigationObservation
    from server.tensor_titans_irrigation_environment import TensorTitansIrrigationEnvironment


# Create the app with web interface and README integration
app = create_app(
    TensorTitansIrrigationEnvironment,
    TensorTitansIrrigationAction,
    TensorTitansIrrigationObservation,
    env_name="tensor_titans_irrigation",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>CropPulse AI</title>
            <style>
                :root {
                    color-scheme: dark;
                    --bg: #07140d;
                    --panel: rgba(14, 31, 22, 0.92);
                    --border: rgba(136, 255, 154, 0.14);
                    --accent: #8de27a;
                    --accent-2: #5ec46b;
                    --text: #ecfff0;
                    --muted: #a7cfaf;
                }
                * { box-sizing: border-box; }
                body {
                    margin: 0;
                    min-height: 100vh;
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                    color: var(--text);
                    background:
                        radial-gradient(circle at top, rgba(141, 226, 122, 0.18), transparent 28%),
                        linear-gradient(180deg, #08130d 0%, #07140d 45%, #050d08 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 24px;
                }
                .card {
                    width: min(880px, 100%);
                    background: var(--panel);
                    border: 1px solid var(--border);
                    border-radius: 24px;
                    padding: 32px;
                    box-shadow: 0 28px 80px rgba(0, 0, 0, 0.35);
                }
                h1 { margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.4rem); }
                .tagline { margin: 0 0 16px; color: var(--accent); font-style: italic; font-size: 1.05rem; }
                .intro { margin: 0 0 24px; color: var(--muted); line-height: 1.7; }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 16px;
                    margin: 24px 0;
                }
                .panel {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 18px;
                    padding: 18px;
                }
                .panel h2 { margin: 0 0 10px; font-size: 1rem; color: var(--accent); }
                .panel p, .panel li { color: var(--muted); line-height: 1.6; margin: 0; }
                ul { padding-left: 18px; margin: 0; }
                .links { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 24px; }
                a {
                    color: #031008;
                    background: linear-gradient(135deg, var(--accent), var(--accent-2));
                    text-decoration: none;
                    padding: 12px 16px;
                    border-radius: 999px;
                    font-weight: 700;
                }
                code { color: var(--text); }
            </style>
        </head>
        <body>
            <main class=\"card\">
                <h1>CropPulse AI</h1>
                <p class=\"tagline\">Where every drop listens to your crops.</p>
                <p class=\"intro\">CropPulse AI is an OpenEnv-compatible irrigation intelligence environment built for evaluating agents on realistic water allocation, crop health, and weather-aware irrigation decisions.</p>
                <section class=\"grid\">
                    <article class=\"panel\">
                        <h2>Available tasks</h2>
                        <ul>
                            <li><code>easy</code>: stable weather and simpler water balancing</li>
                            <li><code>medium</code>: tighter budget and more uncertainty</li>
                            <li><code>hard</code>: aggressive weather shifts and stricter tradeoffs</li>
                        </ul>
                    </article>
                    <article class=\"panel\">
                        <h2>Core endpoints</h2>
                        <ul>
                            <li><code>POST /reset</code></li>
                            <li><code>POST /step</code></li>
                            <li><code>GET /state</code></li>
                            <li><code>GET /schema</code></li>
                            <li><code>GET /healthz</code></li>
                        </ul>
                    </article>
                    <article class=\"panel\">
                        <h2>Submission readiness</h2>
                        <p>Deterministic grading, three tasks, typed models, Docker deployment, and baseline inference are all included in this Space.</p>
                    </article>
                </section>
                <div class=\"links\">
                    <a href=\"/docs\">Open API Docs</a>
                    <a href=\"/schema\">View Schema</a>
                    <a href=\"/healthz\">Health Check</a>
                </div>
            </main>
        </body>
        </html>
        """
    )


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "CropPulse AI"}


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m tensor_titans_irrigation.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn tensor_titans_irrigation.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
