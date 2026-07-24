"""Cenarios EXTRAS da vigia de area: situacoes realistas e adversariais.

Complementa test_cenarios_vigia.py com casos de operacao logistica de verdade:
recolocacao, oclusao parcial, muitas caixas, formatos extremos, deriva de
iluminacao, pessoa/mao passando e caixa na borda da area.

Mesmas convencoes do arquivo base:
- fundo cinza uniforme com ruido leve; caixas = retangulos de tom papelao.
- "estabilizar" = repetir o MESMO frame ate passar stable_needed (sem movimento).
- Detector = None (ZoneMonitor puro): isola a logica de manchas fisicas.

Cada teste diz no docstring se e comportamento que JA funciona (OK) ou uma META
(marcado @pytest.mark.xfail) que mapeia uma limitacao atual do algoritmo.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1] / "04_backend_api_contagem"
sys.path.insert(0, str(BACKEND_DIR))

from app.zone import ZoneMonitor  # noqa: E402

H, W = 480, 640
BG = 120        # fundo cinza
BOX = 195       # tom papelao (contraste > diff_threshold)


def _bg_frame() -> np.ndarray:
    f = np.full((H, W, 3), BG, dtype=np.uint8)
    f = np.clip(f.astype(np.int16) + np.random.randint(-4, 4, f.shape), 0, 255).astype(np.uint8)
    return f


def _put_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, tone: int = BOX) -> None:
    frame[y1:y2, x1:x2] = tone
    frame[y1:y1 + 2, x1:x2] = tone - 40   # aresta superior
    frame[y2 - 2:y2, x1:x2] = tone - 40   # aresta inferior


def _new_zone() -> ZoneMonitor:
    z = ZoneMonitor()
    z.set_roi(0.0, 0.0, 1.0, 1.0)
    return z


def _settle(zone: ZoneMonitor, frame: np.ndarray, n: int = 20) -> dict:
    res = {}
    for _ in range(n):
        res = zone.process(frame)
    return res


def _calibrate_empty(zone: ZoneMonitor) -> None:
    base = _bg_frame()
    for _ in range(20):
        zone.process(base)


# ---------------------------------------------------------------- OK esperados

def test_recolocar_caixa_diferente_soma_total():
    """A colocada, removida, e OUTRA caixa colocada: total de sessao = 2. (OK)

    Fluxo real: operador poe uma caixa, retira, depois deixa outra no mesmo
    lugar. O total acumulado da sessao deve refletir as duas colocacoes.
    """
    z = _new_zone()
    _calibrate_empty(z)

    fa = _bg_frame()
    _put_box(fa, 260, 180, 400, 320)
    _settle(z, fa)                       # A -> total 1

    res_vazio = _settle(z, _bg_frame())  # remove A -> visivel 0, total continua 1
    assert res_vazio["visible"] == 0, res_vazio
    assert res_vazio["count"] == 1, res_vazio

    fb = _bg_frame()
    _put_box(fb, 100, 200, 240, 340)     # B em outro lugar
    res = _settle(z, fb)
    assert res["count"] == 2, res
    assert res["visible"] == 1, res


def test_cinco_caixas_separadas():
    """5 caixas bem separadas devem contar 5. (OK)"""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 20, 190, 110, 300)
    _put_box(f, 150, 190, 240, 300)
    _put_box(f, 280, 190, 370, 300)
    _put_box(f, 410, 190, 500, 300)
    _put_box(f, 540, 190, 630, 300)
    res = _settle(z, f)
    assert res["count"] == 5, res


def test_quatro_caixas_em_grade():
    """4 caixas em 2x2, bem espacadas, devem contar 4. (OK)"""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 80, 60, 260, 200)
    _put_box(f, 380, 60, 560, 200)
    _put_box(f, 80, 280, 260, 420)
    _put_box(f, 380, 280, 560, 420)
    res = _settle(z, f)
    assert res["count"] == 4, res


def test_caixa_alta_estreita():
    """Caixa em pe (alta e estreita) conta 1. (OK)"""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 300, 90, 350, 400)       # 50 x 310
    res = _settle(z, f)
    assert res["count"] == 1, res


def test_caixa_larga_baixa():
    """Caixa deitada (larga e baixa) conta 1. (OK)"""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 120, 220, 520, 275)      # 400 x 55
    res = _settle(z, f)
    assert res["count"] == 1, res


def test_deriva_leve_iluminacao_nao_conta_fantasma():
    """Deriva leve de brilho global (abaixo do limiar) NAO vira caixa. (OK)

    A calibracao guarda o fundo; uma variacao global pequena de iluminacao
    (ex.: nuvem/lampada) fica abaixo de diff_threshold e nao deve contar.
    """
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame().astype(np.int16) + 18   # +18 < diff_threshold(30)
    f = np.clip(f, 0, 255).astype(np.uint8)
    res = _settle(z, f)
    assert res["count"] == 0, res
    assert res["visible"] == 0, res


def test_pessoa_grande_passando_nao_conta():
    """Pessoa cruzando a area (objeto grande, movimento claro) NAO conta. (OK)

    O objeto so aparece em frames em MOVIMENTO (posicoes diferentes) e some.
    Como ocupa parte grande da ROI, cada passo ultrapassa motion_threshold, a
    cena nunca estabiliza com ele presente e nada e confirmado.
    """
    z = _new_zone()
    _calibrate_empty(z)

    # pessoa (retangulo grande escuro) atravessa: cada frame numa posicao
    for x in (20, 150, 280, 410):
        fm = _bg_frame()
        _put_box(fm, x, 120, x + 250, 400, tone=70)
        z.process(fm)

    res = _settle(z, _bg_frame())        # ja saiu; area vazia
    assert res["count"] == 0, res
    assert res["visible"] == 0, res


def test_pessoa_passa_e_deixa_uma_caixa():
    """Pessoa cruza (movimento) e deixa 1 caixa que fica: total = 1. (OK)

    Fluxo real: operador entra andando, larga a caixa e sai. Conta so a caixa.
    """
    z = _new_zone()
    _calibrate_empty(z)

    for x in (40, 200, 360):             # pessoa entrando (movimento)
        fm = _bg_frame()
        _put_box(fm, x, 120, x + 80, 340, tone=70)
        z.process(fm)

    f = _bg_frame()                      # pessoa saiu, caixa fica parada
    _put_box(f, 280, 200, 420, 340)
    res = _settle(z, f)
    assert res["count"] == 1, res


def test_reset_total_zera_sessao():
    """reset_total() zera o acumulado mesmo com caixa ainda visivel. (OK)"""
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 260, 180, 400, 320)
    _settle(z, f)
    z.reset_total()
    res = _settle(z, f)                  # mesma caixa continua na area
    assert res["count"] == 0, res
    assert res["visible"] == 1, res


# ---------------------------------------------------------------- METAS (xfail)

@pytest.mark.xfail(reason="oclusao parcial sem aresta funde as duas manchas em uma; "
                          "so 1 objeto e segmentado, entao o total da sessao nao sobe para 2")
def test_oclusao_parcial_b_cobre_metade_de_a():
    """META: B cobre metade de A (encostadas/sobrepostas): total = 2.

    Duas caixas fisicas distintas; a oclusao parcial deveria manter a contagem
    da sessao em 2 (A foi colocada, depois B).
    """
    z = _new_zone()
    _calibrate_empty(z)
    fa = _bg_frame()
    _put_box(fa, 200, 180, 360, 340)
    _settle(z, fa)                       # A -> 1
    fb = _bg_frame()
    _put_box(fb, 200, 180, 360, 340)     # A continua
    _put_box(fb, 300, 150, 460, 320)     # B sobrepoe metade de A
    res = _settle(z, fb)
    assert res["count"] == 2, res


def test_caixa_colada_na_borda_esquerda():
    """Caixa encostada na borda esquerda da area conta 1. (OK)

    Mesmo com x1 == 0 (colada na fronteira da ROI, onde o watershed marca a
    borda), o objeto e segmentado corretamente.
    """
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame()
    _put_box(f, 0, 170, 120, 330)        # x1 == 0 (borda da ROI)
    res = _settle(z, f)
    assert res["count"] == 1, res


@pytest.mark.xfail(reason="objeto pequeno cruzando area grande gera mean-absdiff abaixo de "
                          "motion_threshold; o guarda de movimento nao dispara, a cena e "
                          "tratada como estavel e o transiente e contado como caixa fantasma")
def test_objeto_pequeno_cruzando_e_contado_como_fantasma():
    """META: objeto pequeno atravessando a ROI grande NAO deveria contar.

    Limitacao conhecida: o movimento e medido pela media do absdiff sobre TODA
    a area; um objeto pequeno (ex.: mao, empilhadeira ao fundo) desloca poucos
    pixels em relacao ao total e fica sob o limiar, sendo confirmado como caixa.
    ROI mais justa em torno da zona de deposito reduz esse falso positivo.
    """
    z = _new_zone()
    _calibrate_empty(z)
    for x in (60, 180, 300, 420, 540):
        fm = _bg_frame()
        _put_box(fm, x, 200, x + 70, 320, tone=70)
        z.process(fm)
    res = _settle(z, _bg_frame())        # objeto ja saiu; area vazia
    assert res["count"] == 0, res


@pytest.mark.xfail(reason="mudanca global forte de iluminacao (acima do limiar) mancha "
                          "toda a ROI e e contada como caixa fantasma")
def test_deriva_forte_iluminacao_conta_fantasma():
    """META: deriva FORTE de brilho global nao deveria virar caixa fantasma.

    Limitacao conhecida: sem recalibrar, um salto grande de iluminacao
    (ex.: alguem acende as luzes do galpao) ultrapassa diff_threshold e a
    area inteira vira uma unica mancha contada como objeto.
    """
    z = _new_zone()
    _calibrate_empty(z)
    f = _bg_frame().astype(np.int16) + 45   # +45 > diff_threshold(30)
    f = np.clip(f, 0, 255).astype(np.uint8)
    res = _settle(z, f)
    assert res["count"] == 0, res
