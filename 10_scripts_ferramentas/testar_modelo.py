"""Testa o detector treinado em imagens e salva o resultado anotado.

Usa o YoloDetector de producao (que prioriza caixas_custom.pt), entao mede
exatamente o que o backend vai enxergar.

Uso:
    # Testa em imagens de validacao sinteticas (amostra aleatoria):
    python 10_scripts_ferramentas/testar_modelo.py

    # Testa numa foto real (recomendado para medir transferencia):
    python 10_scripts_ferramentas/testar_modelo.py --img caminho/para/foto.jpg
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "04_backend_api_contagem"))

from app.yolo_detector import YoloDetector  # noqa: E402

VAL_DIR = Path.home() / ".contador_caixas_ia" / "datasets" / "caixas_sinteticas" / "images" / "val"
SCRATCH = REPO_ROOT / "09_testes_validacao" / "saidas"


def _imread(path: Path) -> np.ndarray | None:
    # Robusto a acento no caminho (Windows).
    data = np.frombuffer(path.read_bytes(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite(path: Path, img: np.ndarray) -> None:
    # cv2.imwrite falha em path com acento no Windows: usa imencode+write_bytes.
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        path.write_bytes(buf.tobytes())


def _annotate(img: np.ndarray, dets: list[dict], kind: str) -> np.ndarray:
    for d in dets:
        cv2.rectangle(img, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]), (0, 220, 255), 2)
        cv2.putText(img, f"{d['label']} {d['conf']:.2f}", (d["x"], max(14, d["y"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    cv2.rectangle(img, (6, 6), (300, 40), (0, 0, 0), -1)
    cv2.putText(img, f"IA({kind}): {len(dets)} caixas", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", help="Foto para testar (ex.: foto real de caixas)")
    ap.add_argument("--n", type=int, default=4, help="Qtd de imagens de val a amostrar")
    args = ap.parse_args()

    det = YoloDetector()
    if not det.load():
        print("Detector nao carregou (ultralytics/modelo indisponivel).")
        sys.exit(1)
    print(f"Modelo ativo: {det.kind}")

    SCRATCH.mkdir(parents=True, exist_ok=True)

    if args.img:
        img = _imread(Path(args.img))
        if img is None:
            print(f"Nao consegui abrir a imagem: {args.img}")
            sys.exit(1)
        dets = det.detect(img)
        out = _annotate(img, dets, det.kind or "?")
        dest = SCRATCH / "teste_modelo_foto.jpg"
        _imwrite(dest, out)
        print(f"{len(dets)} caixas detectadas. Resultado -> {dest}")
        return

    imgs = sorted(VAL_DIR.glob("*.jpg"))
    if not imgs:
        print(f"Sem imagens de val em {VAL_DIR}. Rode gerar_dataset_sintetico.py.")
        sys.exit(1)
    picks = random.sample(imgs, min(args.n, len(imgs)))
    tiles = []
    total = 0
    for p in picks:
        img = _imread(p)
        dets = det.detect(img)
        total += len(dets)
        tiles.append(_annotate(img, dets, det.kind or "?"))
    # Mosaico 2xN/2.
    half = (len(tiles) + 1) // 2
    row1 = np.hstack(tiles[:half])
    row2 = np.hstack(tiles[half:]) if tiles[half:] else None
    mosaic = np.vstack([row1, row2]) if row2 is not None and row2.shape[1] == row1.shape[1] else row1
    dest = SCRATCH / "teste_modelo_val.jpg"
    _imwrite(dest, mosaic)
    print(f"{total} caixas em {len(picks)} imagens. Preview -> {dest}")


if __name__ == "__main__":
    main()
