from pathlib import Path
import sys

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "04_backend_api_contagem"
sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionStore  # noqa: E402
from app.live_counter import LiveCounterService  # noqa: E402
import app.live_counter as live_counter_module  # noqa: E402


class _ClosedCapture:
    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        return None


class _FakeCv2:
    @staticmethod
    def VideoCapture(_source):
        return _ClosedCapture()


@pytest.fixture()
def service(tmp_path: Path) -> LiveCounterService:
    store = SessionStore(tmp_path / "contador.sqlite3")
    store.init_db()
    return LiveCounterService(store=store)


def test_failed_camera_start_closes_session(monkeypatch: pytest.MonkeyPatch, service: LiveCounterService) -> None:
    monkeypatch.setattr(live_counter_module, "require_cv2", lambda: _FakeCv2())

    service.start("camera-invalida")
    assert service._thread is not None
    service._thread.join(timeout=2)

    sessions = service.store.list_sessions(limit=5)

    assert service.status()["running"] is False
    assert service.status()["error"] == "Nao foi possivel abrir a camera: camera-invalida"
    assert sessions[0]["status"] == "failed"
    assert sessions[0]["ended_at"] is not None
