from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "05_ia_visao_computacional" / "modelos"

# Vocabulario de objetos que queremos contar (usado pelo modelo open-vocabulary).
DEFAULT_VOCAB = ["box", "cardboard box", "carton", "package", "parcel", "product"]


class YoloDetector:
    """Detector de objetos por IA (Ultralytics YOLO).

    Ordem de preferencia do modelo:
    1. Modelo customizado treinado com os produtos do usuario (melhor precisao);
    2. Modelo open-vocabulary (YOLO-World) que detecta 'caixa/produto' por texto;
    3. Modelo COCO generico (fallback).
    """

    def __init__(self, conf: float = 0.25) -> None:
        self.conf = conf
        self._model = None
        self._kind: str | None = None
        self._names: dict[int, str] = {}

    @property
    def kind(self) -> str | None:
        return self._kind

    def available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from ultralytics import YOLO
        except Exception:
            return False

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        custom = MODELS_DIR / "caixas_custom.pt"

        try:
            if custom.exists():
                self._model = YOLO(str(custom))
                self._kind = "custom"
            else:
                # Open-vocabulary: detecta caixas/produtos por texto, sem treinar.
                self._model = YOLO("yolov8s-worldv2.pt")
                try:
                    self._model.set_classes(DEFAULT_VOCAB)
                except Exception:
                    pass
                self._kind = "open-vocab"
            self._names = dict(self._model.names) if hasattr(self._model, "names") else {}
            return True
        except Exception:
            # Fallback para COCO generico.
            try:
                self._model = YOLO("yolo11n.pt")
                self._kind = "coco"
                self._names = dict(self._model.names)
                return True
            except Exception:
                self._model = None
                return False

    def detect(self, frame) -> list[dict[str, Any]]:
        """Retorna [{x,y,w,h,label,conf}] dos objetos detectados (ignora pessoas)."""
        if self._model is None and not self.load():
            return []

        results = self._model.predict(frame, conf=self.conf, verbose=False)
        out: list[dict[str, Any]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                label = self._names.get(cls, str(cls))
                if label == "person":
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append(
                    {
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(x2 - x1),
                        "h": int(y2 - y1),
                        "label": label,
                        "conf": float(b.conf[0]) if b.conf is not None else 0.0,
                    }
                )
        return out
