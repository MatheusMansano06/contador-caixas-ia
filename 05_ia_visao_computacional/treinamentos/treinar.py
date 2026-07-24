"""Treina o detector de caixas (Ultralytics YOLO) e publica caixas_custom.pt.

Fluxo:
  1. Garante o dataset (roda o gerador sintetico se ainda nao existir).
  2. Treina a partir do yolo11n.pt (nano, leve para CPU).
  3. Copia o melhor peso para 05_ia_visao_computacional/modelos/caixas_custom.pt,
     que o YoloDetector prioriza automaticamente em producao.

Uso:
    python 05_ia_visao_computacional/treinamentos/treinar.py --epochs 30 --imgsz 416

Para acelerar em maquina fraca: --epochs 15 --imgsz 320 --batch 8
Com GPU: adicione --device 0
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Dataset em caminho ASCII (cv2.imread do ultralytics falha com acento no Windows).
DATASET_DIR = Path.home() / ".contador_caixas_ia" / "datasets" / "caixas_sinteticas"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = REPO_ROOT / "05_ia_visao_computacional" / "modelos"
# Runs tambem em ASCII para evitar problemas de escrita/leitura do ultralytics.
RUNS_DIR = Path.home() / ".contador_caixas_ia" / "runs"
GERADOR = REPO_ROOT / "10_scripts_ferramentas" / "gerar_dataset_sintetico.py"


def _ensure_dataset(train: int, val: int) -> None:
    if DATA_YAML.exists():
        print(f"Dataset encontrado: {DATA_YAML}")
        return
    print("Dataset ausente. Gerando cenas sinteticas...")
    subprocess.run(
        [sys.executable, str(GERADOR), "--train", str(train), "--val", str(val),
         "--dataset-dir", str(DATASET_DIR)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--train-n", type=int, default=300)
    ap.add_argument("--val-n", type=int, default=60)
    args = ap.parse_args()

    _ensure_dataset(args.train_n, args.val_n)

    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        print(f"Ultralytics indisponivel: {exc}\nInstale: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.base)
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name="caixas",
        exist_ok=True,
        verbose=True,
    )

    # Localiza o melhor peso e publica como modelo custom de producao.
    save_dir = Path(getattr(results, "save_dir", RUNS_DIR / "caixas"))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"best.pt nao encontrado em {best}")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "caixas_custom.pt"
    shutil.copyfile(best, dest)
    print(f"\nModelo publicado -> {dest}")
    print("O backend agora usa este modelo automaticamente (prioridade 'custom').")


if __name__ == "__main__":
    main()
