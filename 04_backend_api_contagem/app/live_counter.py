from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

from .counter import CentroidTracker, CountLine, LineCrossingCounter
from .database import SessionStore
from .vision import MotionBoxDetector, draw_overlay, require_cv2


class LiveCounterService:
    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.line = CountLine()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_jpeg: bytes | None = None
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

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**self._status, "line": asdict(self.line)}

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._last_jpeg

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
