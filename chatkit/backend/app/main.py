"""FastAPI entrypoint for the ChatKit starter backend."""

from __future__ import annotations

import time as _time

from chatkit.server import StreamingResult
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .investigation.shell_session import _registry as _shell_registry
from .server import StarterChatServer

app = FastAPI(title="ChatKit Starter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatkit_server = StarterChatServer()


@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    """Proxy the ChatKit web component payload to the server implementation."""
    payload = await request.body()
    result = await chatkit_server.process(payload, {"request": request})

    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "json"):
        return Response(content=result.json, media_type="application/json")
    return JSONResponse(result)


@app.get("/chatkit/session/{thread_id}")
async def session_state(thread_id: str) -> Response:
    """Return persistent shell session state for a thread (used by SessionStateBar)."""
    shell = _shell_registry.get(thread_id)
    if not shell:
        return JSONResponse({"alive": False, "cwd": None, "idle_secs": None})
    return JSONResponse({
        "alive": shell.is_alive(),
        "cwd": shell.last_cwd,
        "idle_secs": int(_time.monotonic() - shell._last_used),
    })


@app.get("/chatkit")
async def chatkit_endpoint_info() -> Response:
    """Informational endpoint for accidental browser GETs; chat requests use POST."""
    return JSONResponse(
        {
            "ok": True,
            "message": "ChatKit endpoint is available. Send POST /chatkit for chat requests.",
        }
    )


@app.put("/chatkit/uploads/{attachment_id}")
async def upload_attachment(attachment_id: str, request: Request) -> Response:
    context = {"request": request}
    try:
        await chatkit_server.store.load_attachment(attachment_id, context=context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Unknown attachment id") from exc

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload payload")

    await chatkit_server.save_attachment_payload(attachment_id, payload)
    return JSONResponse({"ok": True})


@app.get("/chatkit/uploads/{attachment_id}")
async def fetch_attachment(attachment_id: str, request: Request) -> Response:
    context = {"request": request}
    try:
        attachment = await chatkit_server.store.load_attachment(attachment_id, context=context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Unknown attachment id") from exc

    try:
        payload = await chatkit_server.read_attachment_payload(attachment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Attachment payload not found") from exc

    return Response(
        content=payload,
        media_type=attachment.mime_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )
