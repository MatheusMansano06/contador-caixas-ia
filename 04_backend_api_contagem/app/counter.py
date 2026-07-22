from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    area: float

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Track:
    track_id: int
    centroid: tuple[int, int]
    previous_centroid: tuple[int, int] | None = None
    missed_frames: int = 0
    counted: bool = False


@dataclass(frozen=True)
class CountLine:
    x1: float = 0.10
    y1: float = 0.55
    x2: float = 0.90
    y2: float = 0.55

    def as_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x1 * frame_width),
            int(self.y1 * frame_height),
            int(self.x2 * frame_width),
            int(self.y2 * frame_height),
        )

    def side(self, point: tuple[int, int], frame_width: int, frame_height: int) -> float:
        x1, y1, x2, y2 = self.as_pixels(frame_width, frame_height)
        px, py = point
        return float((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))


@dataclass(frozen=True)
class CountEvent:
    track_id: int
    direction: str
    centroid: tuple[int, int]
    total: int


class CentroidTracker:
    def __init__(self, max_distance: float = 90.0, max_missed_frames: int = 12) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: Iterable[Detection]) -> list[Track]:
        detections = list(detections)
        unmatched_detection_indexes = set(range(len(detections)))
        unmatched_track_ids = set(self.tracks.keys())
        assignments: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                distance = hypot(
                    track.centroid[0] - detection.centroid[0],
                    track.centroid[1] - detection.centroid[1],
                )
                if distance <= self.max_distance:
                    assignments.append((distance, track_id, index))

        for _, track_id, detection_index in sorted(assignments, key=lambda item: item[0]):
            if track_id not in unmatched_track_ids or detection_index not in unmatched_detection_indexes:
                continue

            track = self.tracks[track_id]
            track.previous_centroid = track.centroid
            track.centroid = detections[detection_index].centroid
            track.missed_frames = 0
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indexes.remove(detection_index)

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self.tracks[track_id]

        for detection_index in unmatched_detection_indexes:
            detection = detections[detection_index]
            self.tracks[self._next_id] = Track(
                track_id=self._next_id,
                centroid=detection.centroid,
            )
            self._next_id += 1

        return list(self.tracks.values())


class LineCrossingCounter:
    def __init__(self, line: CountLine | None = None) -> None:
        self.line = line or CountLine()
        self.total = 0

    def reset(self) -> None:
        self.total = 0

    def update_line(self, line: CountLine) -> None:
        self.line = line

    def collect_events(
        self,
        tracks: Iterable[Track],
        frame_width: int,
        frame_height: int,
    ) -> list[CountEvent]:
        events: list[CountEvent] = []

        for track in tracks:
            if track.counted or track.previous_centroid is None:
                continue

            previous_side = self.line.side(track.previous_centroid, frame_width, frame_height)
            current_side = self.line.side(track.centroid, frame_width, frame_height)

            if previous_side == 0 or current_side == 0 or previous_side * current_side > 0:
                continue

            self.total += 1
            track.counted = True
            direction = "positivo" if current_side > previous_side else "negativo"
            events.append(
                CountEvent(
                    track_id=track.track_id,
                    direction=direction,
                    centroid=track.centroid,
                    total=self.total,
                )
            )

        return events


def normalize_line_payload(payload: dict[str, float]) -> CountLine:
    values = {
        key: max(0.0, min(1.0, float(payload.get(key, default))))
        for key, default in {"x1": 0.10, "y1": 0.55, "x2": 0.90, "y2": 0.55}.items()
    }
    return CountLine(**values)
