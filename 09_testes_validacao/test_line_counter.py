from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1] / "04_backend_api_contagem"
sys.path.insert(0, str(BACKEND_DIR))

from app.counter import CountLine, LineCrossingCounter, Track  # noqa: E402


def test_counts_when_track_crosses_horizontal_line() -> None:
    counter = LineCrossingCounter(CountLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5))
    track = Track(track_id=7, previous_centroid=(50, 40), centroid=(50, 60), age=3)

    events = counter.collect_events([track], frame_width=100, frame_height=100)

    assert counter.total == 1
    assert len(events) == 1
    assert events[0].track_id == 7
    assert events[0].total == 1


def test_does_not_count_same_track_twice() -> None:
    counter = LineCrossingCounter(CountLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5))
    track = Track(track_id=7, previous_centroid=(50, 40), centroid=(50, 60), age=3)

    counter.collect_events([track], frame_width=100, frame_height=100)
    track.previous_centroid = (50, 60)
    track.centroid = (50, 40)
    events = counter.collect_events([track], frame_width=100, frame_height=100)

    assert counter.total == 1
    assert events == []


def test_ignores_track_that_does_not_cross_line() -> None:
    counter = LineCrossingCounter(CountLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5))
    track = Track(track_id=3, previous_centroid=(20, 20), centroid=(80, 30), age=3)

    events = counter.collect_events([track], frame_width=100, frame_height=100)

    assert counter.total == 0
    assert events == []


def test_ignores_track_younger_than_minimum_age() -> None:
    counter = LineCrossingCounter(CountLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5))
    track = Track(track_id=9, previous_centroid=(50, 40), centroid=(50, 60), age=1)

    events = counter.collect_events([track], frame_width=100, frame_height=100)

    assert counter.total == 0
    assert events == []
