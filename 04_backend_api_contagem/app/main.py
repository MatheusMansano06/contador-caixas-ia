from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .counter import normalize_line_payload
from .database import SessionStore
from .live_counter import LiveCounterService


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "03_frontend_mobile_camera"

store = SessionStore()
service = LiveCounterService(store=store)

app = FastAPI(title="Contador de Caixas IA", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


class StartRequest(BaseModel):
    source: str = "0"


class LineRequest(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


@app.on_event("startup")
def startup() -> None:
    store.init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    return service.status()


@app.post("/api/start")
def start(payload: StartRequest) -> dict:
    return service.start(source=payload.source.strip() or "0")


@app.post("/api/stop")
def stop() -> dict:
    return service.stop()


@app.post("/api/line")
def set_line(payload: LineRequest) -> dict:
    line = normalize_line_payload(payload.dict())
    return service.set_line(line)


@app.get("/api/sessions")
def sessions(limit: int = 20) -> list[dict]:
    return store.list_sessions(limit=limit)


@app.get("/api/events/{session_id}")
def events(session_id: int, limit: int = 100) -> list[dict]:
    return store.list_events(session_id=session_id, limit=limit)


@app.get("/api/stream")
def stream() -> StreamingResponse:
    if not service.status()["running"]:
        raise HTTPException(status_code=409, detail="Nenhuma sessao ativa.")
    return StreamingResponse(_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


def _frame_generator() -> Generator[bytes, None, None]:
    while service.status()["running"]:
        frame = service.snapshot()
        if frame is None:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)
