from __future__ import annotations

from typing import Iterable

from .counter import CountLine, Detection, Track


class OpenCVNotAvailable(RuntimeError):
    pass


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise OpenCVNotAvailable(
            "OpenCV nao esta instalado. Rode `pip install -r 04_backend_api_contagem/requirements.txt`."
        ) from exc
    return cv2


class MotionBoxDetector:
    """Detector local inicial baseado em movimento e contornos."""

    def __init__(self, min_area: int = 1600) -> None:
        cv2 = require_cv2()
        self.cv2 = cv2
        self.min_area = min_area
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=250,
            varThreshold=36,
            detectShadows=True,
        )

    def detect(self, frame) -> list[Detection]:
        cv2 = self.cv2
        mask = self._subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(Detection(x=x, y=y, width=width, height=height, area=area))

        return detections


def draw_overlay(
    frame,
    detections: Iterable[Detection],
    tracks: Iterable[Track],
    line: CountLine,
    total: int,
    status_text: str,
):
    cv2 = require_cv2()
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = line.as_pixels(frame_width, frame_height)
    output = frame.copy()

    cv2.line(output, (x1, y1), (x2, y2), (42, 220, 170), 3)

    for detection in detections:
        cv2.rectangle(
            output,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            (255, 190, 80),
            2,
        )

    for track in tracks:
        cx, cy = track.centroid
        color = (120, 220, 255) if not track.counted else (120, 255, 140)
        cv2.circle(output, (cx, cy), 5, color, -1)
        cv2.putText(
            output,
            f"ID {track.track_id}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (12, 12), (320, 82), (18, 24, 32), -1)
    cv2.putText(output, f"Total: {total}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(output, status_text, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 205, 220), 1)
    return output
