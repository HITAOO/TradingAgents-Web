"""
FastAPI backend for TradingAgents.

Endpoints:
  GET  /health                    → liveness check
  GET  /api/providers             → list of available LLM providers
  GET  /api/models?provider=xxx   → model list for a provider
  POST /api/analyze/stream        → SSE stream of analysis events
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS, get_model_options
from web.config_builder import PROVIDER_DISPLAY_NAMES
from web.models import AnalysisRequest
from web.stream_handler import AnalysisStreamHandler

app = FastAPI(
    title="TradingAgents API",
    description="Multi-Agent LLM Financial Trading Analysis",
    version="1.0.0",
)

_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@app.get("/api/providers")
async def list_providers():
    return [{"label": name, "value": key} for name, key in PROVIDER_DISPLAY_NAMES]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models(provider: str):
    provider_lower = provider.lower()

    if provider_lower == "openrouter":
        # Try to fetch live; fall back gracefully
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://openrouter.ai/api/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                options = [(m.get("name") or m["id"], m["id"]) for m in models[:50]]
        except Exception:
            options = []
        return {
            "quick": [{"label": l, "value": v} for l, v in options],
            "deep":  [{"label": l, "value": v} for l, v in options],
            "free_entry": True,
        }

    if provider_lower == "azure":
        return {"quick": [], "deep": [], "free_entry": True}

    if provider_lower not in MODEL_OPTIONS:
        return {"quick": [], "deep": [], "free_entry": True}

    quick = [{"label": l, "value": v} for l, v in get_model_options(provider_lower, "quick")]
    deep  = [{"label": l, "value": v} for l, v in get_model_options(provider_lower, "deep")]
    return {"quick": quick, "deep": deep, "free_entry": False}


# ---------------------------------------------------------------------------
# SSE streaming analysis
# ---------------------------------------------------------------------------

@app.post("/api/analyze/stream")
async def analyze_stream(request: AnalysisRequest, http_request: Request):
    """
    Run analysis and stream Server-Sent Events.

    Event format:
      event: <type>
      data: <json>

    Types: agent_status | report_section | message | tool_call | stats | done | error
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def run_sync():
            handler = AnalysisStreamHandler(request)
            try:
                for event in handler.iter_events():
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                import traceback
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "error",
                     "data": {"message": str(exc), "detail": traceback.format_exc()}},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        _executor.submit(run_sync)

        while True:
            if await http_request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Send a keepalive comment so proxies don't close the connection
                yield ": keepalive\n\n"
                continue

            if event is None:
                break

            yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
