"""Bateria de cenarios da vigia de area (contagem de caixas deixadas).

Simula sequencias de frames (camera fixa) alimentando o ZoneMonitor:
calibra area vazia -> coloca/empilha/oculta caixas -> confere a contagem.

Detector = None de proposito: isola a LOGICA de contagem por manchas fisicas
(sem depender do modelo YOLO). Cada cenario da lista do cliente vira um teste.

Convencoes:
- fundo cinza uniforme; caixas = retangulos de tom papelao (alto contraste).
- "estabilizar" = repetir o mesmo frame ate passar stable_needed.
"""
from pathlib import Path
import sys

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1] / "04_backend_api_contagem"
sys.path.insert(0, str(BACKEND_DIR))

from app.zone import ZoneMonitor  # noqa: E402

H, W = 480, 640
BG = 120        # fundo cinza
BOX = 195       # tom papelao (contraste > diff_threshold)


def _bg_frame() -> np.ndarray:
    f = np.full((H, W, 3), BG, dtype=np.uint8)
    # ruido leve para nao ser sinteticamente perfeito
    f = np.clip(f.astype(np.int16) + np.random.randint(-4, 4, f.shape), 0, 255).astype(np.uint8)
    return f


def _put_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, tone: int = BOX) -> None:
    frame[y1:y2, x1:x2] = tone
    # borda mais escura (aresta da caixa)
    frame[y1:y1 + 2, x1:x2] = tone - 40
    frame[y2 - 2:y2, x1:x2] = tone - 40


def _new_zone() -> ZoneMonitor:
    z = ZoneMonitor()          # sem detector -> logica de manchas pura
    z.set_roi(0.0, 0.0, 1.0, 1.0)
    return z


def _settle(zone: ZoneMonitor, frame: np.ndarray, n: int = 20) -> dict:
    # Repete o MESMO frame para nao gerar movimento entre iteracoes (so o 1o,
    # apos uma troca de cena, conta como movimento).
    res = {}
    for _ in range(n):
        res = zone.process(frame)
    return res


def _calibrate_empty(zone: ZoneMonitor) -> None:
    # 1o frame aprende o fundo; repetir estabiliza.
    base = _bg_frame()
    for _ in range(20):
        zone.process(base)


# ---------------------------------------------------------------- cenarios

def test_uma_caixa():
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 260, 180, 400, 320)
    res = _settle(z, f)
    assert res["count"] == 1, res


def test_duas_separadas():
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 80, 180, 220, 320)
    _put_box(f, 420, 180, 560, 320)
    res = _settle(z, f)
    assert res["count"] == 2, res


def test_tres_separadas_tamanhos_variados():
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 40, 200, 130, 300)     # pequena
    _put_box(f, 260, 150, 400, 340)    # grande
    _put_box(f, 500, 210, 600, 300)    # media
    res = _settle(z, f)
    assert res["count"] == 3, res


def test_caixa_pequena_e_grande_juntas_separadas():
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 60, 120, 300, 380)     # bem grande
    _put_box(f, 420, 250, 490, 320)    # bem pequena
    res = _settle(z, f)
    assert res["count"] == 2, res


def test_remover_caixa_zera_contagem_visivel():
    """Sem persistencia de sessao ainda: ao remover, current deve cair."""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 260, 180, 400, 320)
    _settle(z, f)
    res = _settle(z, _bg_frame())
    assert res["visible"] == 0, res


# --- cenarios DIFICEIS (metas; podem falhar no baseline) -------------------

def _seam_v(frame: np.ndarray, x: int, y1: int, y2: int) -> None:
    """Aresta/sombra vertical entre duas caixas encostadas (como no real)."""
    frame[y1:y2, x - 2:x + 2] = 45


def _seam_h(frame: np.ndarray, y: int, x1: int, x2: int) -> None:
    frame[y - 2:y + 2, x1:x2] = 45


def test_duas_grudadas():
    """Duas caixas encostadas (com aresta entre elas) devem contar 2."""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 180, 180, 320, 320)
    _put_box(f, 320, 180, 460, 320)   # encostada na anterior
    _seam_v(f, 320, 180, 320)         # crease/sombra visivel entre elas
    res = _settle(z, f)
    assert res["count"] == 2, res


def test_empilhada_uma_sobre_a_outra():
    """Uma caixa sobre a outra (com aresta horizontal) deve contar 2."""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 240, 240, 420, 360)   # base
    _put_box(f, 260, 120, 400, 240)   # em cima
    _seam_h(f, 240, 260, 400)         # aresta entre as duas
    res = _settle(z, f)
    assert res["count"] == 2, res


def test_oculta_apos_colocada_continua_contando():
    """Coloca A (conta 1), coloca B na frente ocultando A: total da sessao = 2."""
    z = _new_zone()
    _calibrate_empty(z)
    fa = _bg_frame()
    _put_box(fa, 240, 200, 380, 340)
    _settle(z, fa)                     # A colocada -> 1
    fb = _bg_frame()
    _put_box(fb, 200, 160, 420, 360)  # B maior, cobre A
    res = _settle(z, fb)
    assert res["count"] == 2, res      # sessao lembra que 2 foram colocadas
