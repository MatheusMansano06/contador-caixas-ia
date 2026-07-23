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


@app.get("/index-mobile.html")
def index_mobile() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index-mobile.html")


@app.get("/monitor.html")
def monitor() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "monitor.html")


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


@app.post("/api/reset")
def reset() -> dict:
    return service.reset_count()


class ZoneRequest(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


@app.post("/api/zone")
def set_zone(payload: ZoneRequest) -> dict:
    return service.set_zone(payload.x1, payload.y1, payload.x2, payload.y2)


@app.post("/api/zone/clear")
def clear_zone() -> dict:
    return service.clear_zone()


@app.post("/api/zone/calibrate")
def calibrate_zone() -> dict:
    return service.recalibrate_zone()


@app.post("/api/rotate")
def rotate() -> dict:
    return service.cycle_rotation()


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


@app.post("/api/frame")
def process_frame(body: dict) -> dict:
    """Recebe frame base64 do cliente (câmera móvel)."""
    if "data" not in body:
        raise HTTPException(status_code=400, detail="Falta dados de frame (chave 'data')")
    try:
        import base64
        frame_b64 = body["data"]
        if frame_b64.startswith("data:image"):
            frame_b64 = frame_b64.split(",")[1]
        frame_data = base64.b64decode(frame_b64)
        result = service.process_frame(frame_data)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/snapshot")
def snapshot():
    frame = service.snapshot()
    if frame is None:
        raise HTTPException(status_code=404, detail="Sem frame disponivel")
    from fastapi.responses import Response
    return Response(content=frame, media_type="image/jpeg")


@app.get("/api/test-yolo")
def test_yolo(conf: float = 0.15):
    """Roda a IA no ultimo frame cru e retorna a imagem anotada + contagem."""
    from fastapi.responses import Response
    import cv2
    from .yolo_detector import YoloDetector

    raw = service._last_raw
    if raw is None:
        raise HTTPException(status_code=404, detail="Sem frame. Inicie a camera no celular.")

    frame = raw.copy()
    det = YoloDetector(conf=conf)
    if not det.load():
        raise HTTPException(status_code=500, detail="IA nao carregou")
    boxes = det.detect(frame)
    for b in boxes:
        cv2.rectangle(frame, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 220, 255), 3)
        cv2.putText(frame, f"{b['label']} {b['conf']:.2f}", (b["x"], max(20, b["y"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.rectangle(frame, (10, 10), (330, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"IA ({det.kind}): {len(boxes)} objetos", (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    ok, encoded = cv2.imencode(".jpg", frame)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


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
