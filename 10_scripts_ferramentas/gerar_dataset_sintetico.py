"""Gera um dataset sintetico de caixas em formato YOLO.

Compoe caixas de papelao (com textura, sombreamento, fita, sombra projetada,
rotacao/perspectiva e oclusao) sobre fundos procedurais com distratores.
Serve de bootstrap para treinar o detector antes de haver fotos reais.

Uso:
    python 10_scripts_ferramentas/gerar_dataset_sintetico.py --train 300 --val 60

Saida (formato Ultralytics):
    05_ia_visao_computacional/datasets/caixas_sinteticas/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt   (classe x_center y_center w h, normalizado)
        data.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ATENCAO Windows: cv2.imread/imwrite falham com acento no caminho (ex.: "Área de
# Trabalho"). O ultralytics usa cv2.imread para carregar as imagens de treino, entao
# o dataset PRECISA ficar num caminho ASCII. Default = pasta ASCII no home do usuario.
# Sobreponha com --dataset-dir se quiser outro local.
ASCII_BASE = Path.home() / ".contador_caixas_ia" / "datasets" / "caixas_sinteticas"
DATASET_DIR = ASCII_BASE

W, H = 640, 480


def _procedural_background() -> np.ndarray:
    """Fundo tipo piso/bancada: cor base + gradiente + ruido + textura leve."""
    base = np.random.randint(60, 190, size=3)
    bg = np.full((H, W, 3), base, dtype=np.uint8)

    # Gradiente de iluminacao (horizontal ou vertical).
    if random.random() < 0.5:
        grad = np.tile(np.linspace(-40, 40, W, dtype=np.float32), (H, 1))  # (H,W)
    else:
        grad = np.tile(np.linspace(-40, 40, H, dtype=np.float32)[:, None], (1, W))  # (H,W)
    bg = np.clip(bg.astype(np.float32) + grad[:, :, None], 0, 255).astype(np.uint8)

    # Ruido fino.
    noise = np.random.randint(-18, 18, (H, W, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Linhas de textura (juntas de piso).
    for _ in range(random.randint(0, 4)):
        y = random.randint(0, H - 1)
        cv2.line(bg, (0, y), (W, y), tuple(int(c) for c in np.random.randint(40, 120, 3)), 1)
    return bg


def _add_distractors(img: np.ndarray) -> None:
    """Objetos que NAO sao caixa (reduz falso positivo no treino)."""
    for _ in range(random.randint(0, 3)):
        kind = random.random()
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        if kind < 0.5:
            c = (random.randint(0, W), random.randint(0, H))
            cv2.circle(img, c, random.randint(10, 45), color, -1)
        else:
            p1 = (random.randint(0, W), random.randint(0, H))
            p2 = (p1[0] + random.randint(10, 60), p1[1] + random.randint(4, 14))
            cv2.rectangle(img, p1, p2, color, -1)


def _draw_box(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Desenha uma caixa de papelao e retorna sua bbox (x1,y1,x2,y2) em pixels."""
    bw = random.randint(70, 200)
    bh = random.randint(60, 180)
    cx = random.randint(bw // 2, W - bw // 2)
    cy = random.randint(bh // 2, H - bh // 2)

    # Retangulo base -> rotaciona/deforma levemente (perspectiva).
    rect = np.array([
        [cx - bw / 2, cy - bh / 2],
        [cx + bw / 2, cy - bh / 2],
        [cx + bw / 2, cy + bh / 2],
        [cx - bw / 2, cy + bh / 2],
    ], dtype=np.float32)
    ang = random.uniform(-0.35, 0.35)
    cos, sin = np.cos(ang), np.sin(ang)
    rot = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)
    quad = (rect - center) @ rot.T + center
    quad += np.random.uniform(-6, 6, quad.shape).astype(np.float32)  # perspectiva sutil
    quad_i = quad.astype(np.int32)

    # Sombra projetada.
    shadow = quad_i + np.array([random.randint(4, 12), random.randint(4, 12)])
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, shadow, (30, 30, 30))
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Corpo da caixa: tom papelao + variacao.
    tan = np.array([132, 164, 196], dtype=np.float32)  # BGR
    tan += np.random.uniform(-30, 30, 3)
    tan = np.clip(tan, 40, 235)
    cv2.fillConvexPoly(img, quad_i, tuple(int(c) for c in tan))

    # Textura/ruido e sombreamento dentro da caixa.
    x1, y1 = quad_i[:, 0].min(), quad_i[:, 1].min()
    x2, y2 = quad_i[:, 0].max(), quad_i[:, 1].max()
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    if x2c - x1c > 4 and y2c - y1c > 4:
        roi = img[y1c:y2c, x1c:x2c].astype(np.int16)
        roi += np.random.randint(-14, 14, roi.shape, dtype=np.int16)
        img[y1c:y2c, x1c:x2c] = np.clip(roi, 0, 255).astype(np.uint8)

    # Fita/seam no meio.
    mid_top = ((quad_i[0] + quad_i[1]) / 2).astype(int)
    mid_bot = ((quad_i[3] + quad_i[2]) / 2).astype(int)
    cv2.line(img, tuple(mid_top), tuple(mid_bot),
             tuple(int(c) for c in (tan * 0.8)), 2)
    # Aba superior.
    cv2.line(img, tuple(quad_i[0]), tuple(quad_i[1]),
             tuple(int(c) for c in (tan * 0.7)), 1)

    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(W, x2), min(H, y2)
    if bx2 - bx1 < 12 or by2 - by1 < 12:
        return None
    return bx1, by1, bx2, by2


def _make_scene() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    img = _procedural_background()
    _add_distractors(img)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(random.randint(0, 6)):  # inclui cenas vazias (negativos)
        bb = _draw_box(img)
        if bb is not None:
            boxes.append(bb)
    # Blur global leve para simular camera.
    if random.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, boxes


def _to_yolo_label(bb: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bb
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _generate_split(split: str, n: int) -> None:
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img, boxes = _make_scene()
        stem = f"{split}_{i:05d}"
        # imencode+write_bytes: cv2.imwrite falha em paths com acento no Windows.
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("Falha ao codificar imagem")
        (img_dir / f"{stem}.jpg").write_bytes(buf.tobytes())
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(_to_yolo_label(b) for b in boxes), encoding="utf-8"
        )
    print(f"[{split}] {n} imagens em {img_dir}")


def _write_data_yaml() -> Path:
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['caixa']\n",
        encoding="utf-8",
    )
    print(f"data.yaml -> {yaml_path}")
    return yaml_path


def main() -> None:
    global DATASET_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-dir", default=str(ASCII_BASE),
                    help="Pasta do dataset (use caminho ASCII no Windows)")
    args = ap.parse_args()

    DATASET_DIR = Path(args.dataset_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _generate_split("train", args.train)
    _generate_split("val", args.val)
    _write_data_yaml()
    print("Dataset sintetico pronto.")


if __name__ == "__main__":
    main()
