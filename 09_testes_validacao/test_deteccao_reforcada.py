"""Testes das melhorias de identificacao/contagem em producao."""
from pathlib import Path
import sys

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1] / "04_backend_api_contagem"
sys.path.insert(0, str(BACKEND_DIR))

from app.counter import CentroidTracker, CountLine, Detection, LineCrossingCounter, Track  # noqa: E402
from app.database import SessionStore  # noqa: E402
from app.live_counter import LiveCounterService  # noqa: E402
from app.yolo_detector import _merge_overlapping  # noqa: E402
from app.zone import ZoneMonitor  # noqa: E402


# --- Fusao de deteccoes da mesma caixa fisica ------------------------------

def test_faces_da_mesma_caixa_sao_fundidas_em_uma() -> None:
    dets = [
        {"x": 0, "y": 0, "w": 100, "h": 100, "label": "caixa", "conf": 0.9},
        {"x": 10, "y": 10, "w": 40, "h": 40, "label": "caixa", "conf": 0.7},   # face interna
        {"x": 200, "y": 0, "w": 50, "h": 50, "label": "caixa", "conf": 0.8},   # outra caixa
    ]
    merged = _merge_overlapping(dets, 0.6)
    assert len(merged) == 2


def test_caixas_vizinhas_encostadas_nao_sao_fundidas() -> None:
    dets = [
        {"x": 0, "y": 0, "w": 100, "h": 100, "label": "caixa", "conf": 0.9},
        {"x": 95, "y": 0, "w": 100, "h": 100, "label": "caixa", "conf": 0.9},  # 5px de encosto
    ]
    assert len(_merge_overlapping(dets, 0.6)) == 2


# --- Tracker: previsao de velocidade -------------------------------------

def test_track_prediz_proxima_posicao_pela_velocidade() -> None:
    t = Track(track_id=1, centroid=(100, 50), velocity=(10.0, -4.0))
    assert t.predicted_centroid() == (110, 46)


def test_tracker_mantem_id_com_objeto_em_movimento_rapido() -> None:
    tracker = CentroidTracker(max_distance=40.0)
    # Frame 1: objeto em (10,10)
    tracker.update([Detection(x=5, y=5, width=10, height=10, area=100)])
    # Frame 2: anda +30px -> cria velocidade
    tracker.update([Detection(x=35, y=5, width=10, height=10, area=100)])
    tid = next(iter(tracker.tracks))
    # Frame 3: anda +30px de novo. Distancia crua (30) < max, mas a previsao
    # deve casar com folga, mantendo o MESMO id (sem criar track novo).
    tracker.update([Detection(x=65, y=5, width=10, height=10, area=100)])
    assert len(tracker.tracks) == 1
    assert tid in tracker.tracks


# --- Contagem: guarda anti-tremor ----------------------------------------

def test_caixa_tremendo_sobre_a_linha_nao_conta() -> None:
    counter = LineCrossingCounter(CountLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5), min_travel=6.0)
    # Cruza a linha (49->51) mas move so 2px: e tremor, nao passagem.
    track = Track(track_id=1, previous_centroid=(50, 49), centroid=(50, 51), age=5)
    events = counter.collect_events([track], frame_width=100, frame_height=100)
    assert counter.total == 0
    assert events == []


def test_passagem_real_ainda_conta_com_guarda_ativa() -> None:
    counter = LineCrossingCounter(CountLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5), min_travel=6.0)
    track = Track(track_id=1, previous_centroid=(50, 40), centroid=(50, 60), age=5)
    events = counter.collect_events([track], frame_width=100, frame_height=100)
    assert counter.total == 1
    assert len(events) == 1


# --- Config em tempo real -------------------------------------------------

def test_config_atualiza_e_limita_valores(tmp_path: Path) -> None:
    svc = LiveCounterService(store=SessionStore(db_path=tmp_path / "t.db"))
    out = svc.set_config({"yolo": {"conf": 0.5}, "line": {"min_travel": 12}})
    assert out["yolo"]["conf"] == 0.5
    assert out["line"]["min_travel"] == 12.0
    # Clamp: valores fora do intervalo sao limitados, nao aceitos crus.
    assert svc.set_config({"yolo": {"conf": 9}})["yolo"]["conf"] == 0.95
    assert svc.set_config({"yolo": {"conf": -1}})["yolo"]["conf"] == 0.01


# --- Zone: fallback sem detector -----------------------------------------

def test_uma_caixa_vira_uma_mancha_unica() -> None:
    """Fragmentos da mesma caixa (fita/costura) viram 1 bloco fisico apos dilatacao."""
    zone = ZoneMonitor()
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[40:90, 40:150] = 255   # corpo da caixa
    mask[92:140, 60:130] = 255  # face de baixo, com pequeno vao (seam)
    blobs = zone._merged_blobs(mask)
    assert len(blobs) == 1


def test_duas_caixas_separadas_sao_dois_blocos() -> None:
    zone = ZoneMonitor()
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[40:120, 20:90] = 255     # caixa 1
    mask[40:120, 200:270] = 255   # caixa 2 (bem separada)
    assert len(zone._merged_blobs(mask)) == 2


def test_mancha_so_conta_se_yolo_confirmar() -> None:
    """_overlaps_any confirma que a mancha fisica coincide com uma caixa do YOLO."""
    from app.zone import _overlaps_any
    blob = (50, 50, 40, 40)
    assert _overlaps_any(blob, [(60, 60, 30, 30)]) is True   # sobrepoe caixa
    assert _overlaps_any(blob, [(200, 200, 20, 20)]) is False  # sombra longe, sem caixa


def test_zone_sem_detector_usa_movimento() -> None:
    zone = ZoneMonitor(detector=None)
    zone.set_roi(0.1, 0.1, 0.9, 0.9)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = zone.process(frame)  # primeiro frame: aprende fundo
    assert result is not None
    assert result["detector"] == "movimento"
    assert result["calibrating"] is True
