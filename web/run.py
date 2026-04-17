"""
Entry point for the TradingAgents web frontend.

Usage:
  # Gradio standalone (default, for local development)
  python -m web.run

  # FastAPI server with Gradio mounted at /ui  (for deployment)
  python -m web.run --mode server

  # Gradio with a public share link
  python -m web.run --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env before importing anything that touches the environment
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
load_dotenv(".env.enterprise", override=False)


def run_gradio(host: str, port: int, share: bool) -> None:
    from web.gradio_app import create_demo
    demo = create_demo()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
    )


def run_server(host: str, port: int) -> None:
    """Mount Gradio under /ui and serve both the REST API and the UI together."""
    import gradio as gr
    import uvicorn
    from web.app import app as fastapi_app
    from web.gradio_app import create_demo

    demo = create_demo()
    app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
    print(f"\nTradingAgents server starting at  http://{host}:{port}")
    print(f"  Web UI  →  http://{host}:{port}/ui")
    print(f"  API     →  http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradingAgents Web Frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["gradio", "server"],
        default="gradio",
        help="Launch mode: 'gradio' (standalone UI) or 'server' (FastAPI + Gradio at /ui)",
    )
    parser.add_argument("--host",  default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",  type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link (gradio mode only)")
    args = parser.parse_args()

    if args.mode == "gradio":
        run_gradio(args.host, args.port, args.share)
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()
