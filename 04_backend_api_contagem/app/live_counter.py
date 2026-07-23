from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .counter import CentroidTracker, CountLine, LineCrossingCounter
from .database import SessionStore
from .vision import MotionBoxDetector, draw_overlay, require_cv2
from .zone import ZoneMonitor, draw_zone_overlay


class LiveCounterService:
    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.line = CountLine()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_jpeg: bytes | None = None
        self._last_raw = None  # ultimo frame sem overlay (para testar/rodar IA)
        self._detector: MotionBoxDetector | None = None
        self._tracker: CentroidTracker | None = None
        self._counter: LineCrossingCounter | None = None
        self._zone = ZoneMonitor()
        self._rotation = 0  # 0, 90, 180, 270
        self._status: dict[str, Any] = {
            "running": False,
            "session_id": None,
            "source": None,
            "total": 0,
            "fps": 0.0,
            "frame_width": None,
            "frame_height": None,
            "detections": 0,
            "tracks": 0,
            "error": None,
        }

    def start(self, source: str) -> dict[str, Any]:
        with self._lock:
            if self._status["running"]:
                return self.status()

            self._stop_event.clear()
            self._last_jpeg = None
            self._detector = None
            self._tracker = None
            self._counter = None
            session_id = self.store.create_session(source=source, line_config=asdict(self.line))
            self._status.update(
                {
                    "running": True,
                    "session_id": session_id,
                    "source": source,
                    "total": 0,
                    "fps": 0.0,
                    "detections": 0,
                    "tracks": 0,
                    "error": None,
                }
            )
            # Modo "client": frames vêm do celular via /api/frame, sem thread de captura local.
            if source == "client":
                return self.status()
            self._thread = threading.Thread(target=self._run, args=(source, session_id), daemon=True)
            self._thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            session_id = self._status["session_id"]
            total = int(self._status["total"])
            self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

        with self._lock:
            if session_id is not None:
                self.store.finish_session(int(session_id), total)
            self._status["running"] = False
            return self.status()

    def set_line(self, line: CountLine) -> dict[str, Any]:
        with self._lock:
            self.line = line
            return self.status()

    def reset_count(self) -> dict[str, Any]:
        with self._lock:
            if self._counter is not None:
                self._counter.reset()
            if self._tracker is not None:
                self._tracker.tracks.clear()
            self._zone.reset_total()
            self._status["total"] = 0
            session_id = self._status["session_id"]
            if session_id is not None:
                self.store.update_total(int(session_id), 0)
            return self.status()

    def set_zone(self, x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
        with self._lock:
            self._zone.set_roi(x1, y1, x2, y2)
            self._status["total"] = 0
            return self.status()

    def clear_zone(self) -> dict[str, Any]:
        with self._lock:
            self._zone.clear()
            self._status["total"] = 0
            return self.status()

    def recalibrate_zone(self) -> dict[str, Any]:
        with self._lock:
            self._zone.recalibrate()
            return self.status()

    def cycle_rotation(self) -> dict[str, Any]:
        with self._lock:
            self._rotation = (self._rotation + 90) % 360
            # A imagem mudou de orientacao: a area/fundo precisam recalibrar.
            self._zone.recalibrate()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**self._status, "line": asdict(self.line), "rotation": self._rotation}

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._last_jpeg

    def process_frame(self, frame_data: bytes) -> dict[str, Any]:
        """Processa um frame enviado do cliente (câmera móvel)."""
        cv2 = require_cv2()

        if not self._status["running"]:
            return {"error": "Nenhuma sessão ativa"}

        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"error": "Frame inválido"}

            rotation = self._rotation
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            session_id = self._status["session_id"]
            frame_height, frame_width = frame.shape[:2]
            with self._lock:
                self._last_raw = frame.copy()

            # Modo vigia de area: conta objetos deixados/parados dentro da ROI.
            if self._zone.active:
                with self._lock:
                    result = self._zone.process(frame)
                if result is not None:
                    overlay = draw_zone_overlay(frame, result)
                    _, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    count = int(result["count"])
                    visible = int(result.get("visible", 0))
                    with self._lock:
                        self._last_jpeg = encoded.tobytes()
                        prev_total = self._status["total"]
                        self._status.update(
                            {
                                "total": count,
                                "visible": visible,
                                "frame_width": frame_width,
                                "frame_height": frame_height,
                                "detections": len(result.get("boxes", [])),
                                "tracks": len(result.get("boxes", [])),
                            }
                        )
                    if session_id is not None and count != prev_total:
                        self.store.update_total(session_id, count)
                    return {"total": count, "visible": visible, "tracks": len(result.get("boxes", [])),
                            "frame_width": frame_width, "frame_height": frame_height}

            # Sem area definida: apenas transmite a imagem, sem contar nada.
            cv2.putText(
                frame,
                "Desenhe a area de contagem no computador",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (60, 160, 255),
                2,
            )
            _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            with self._lock:
                self._last_jpeg = encoded.tobytes()
                self._status.update(
                    {
                        "total": 0,
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                        "detections": 0,
                        "tracks": 0,
                    }
                )
            return {"total": 0, "tracks": 0, "frame_width": frame_width, "frame_height": frame_height}
        except Exception as exc:
            return {"error": str(exc)}

    def _run(self, source: str, session_id: int) -> None:
        cv2 = require_cv2()
        capture_source: int | str = int(source) if source.isdigit() else source
        capture = cv2.VideoCapture(capture_source)
        detector = MotionBoxDetector()
        tracker = CentroidTracker()
        counter = LineCrossingCounter(self.line)
        last_tick = time.monotonic()
        frames = 0

        try:
            if not capture.isOpened():
                self._set_error(f"Nao foi possivel abrir a camera: {source}")
                return

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self._set_error("Falha ao ler frame da camera.")
                    time.sleep(0.2)
                    continue

                frame_height, frame_width = frame.shape[:2]
                with self._lock:
                    counter.update_line(self.line)

                detections = detector.detect(frame)
                tracks = tracker.update(detections)
                events = counter.collect_events(tracks, frame_width, frame_height)

                for event in events:
                    self.store.insert_event(
                        session_id=session_id,
                        track_id=event.track_id,
                        direction=event.direction,
                        centroid=event.centroid,
                        total=event.total,
                    )
                    self.store.update_total(session_id, event.total)

                frames += 1
                now = time.monotonic()
                elapsed = now - last_tick
                fps = frames / elapsed if elapsed >= 1 else self._status["fps"]
                if elapsed >= 1:
                    frames = 0
                    last_tick = now

                overlay = draw_overlay(
                    frame=frame,
                    detections=detections,
                    tracks=tracks,
                    line=counter.line,
                    total=counter.total,
                    status_text=f"FPS {fps:.1f} | objetos {len(tracks)}",
                )
                _, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

                with self._lock:
                    self._last_jpeg = encoded.tobytes()
                    self._status.update(
                        {
                            "total": counter.total,
                            "fps": round(float(fps), 1),
                            "frame_width": frame_width,
                            "frame_height": frame_height,
                            "detections": len(detections),
                            "tracks": len(tracks),
                            "error": None,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
        finally:
            capture.release()
            with self._lock:
                self._status["running"] = False

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._status["error"] = message
            self._status["running"] = False
