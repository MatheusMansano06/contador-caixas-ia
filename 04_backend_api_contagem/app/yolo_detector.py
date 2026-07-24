from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "05_ia_visao_computacional" / "modelos"

# Vocabulario de objetos que queremos contar (usado pelo modelo open-vocabulary).
DEFAULT_VOCAB = ["box", "cardboard box", "carton", "package", "parcel", "product"]


def _merge_overlapping(dets: list[dict[str, Any]], thresh: float) -> list[dict[str, Any]]:
    """Funde deteccoes que se sobrepoem fortemente (faces da MESMA caixa fisica).

    Criterio: intersecao / area do menor bbox >= thresh -> mesmo objeto (vira a
    uniao dos dois). Caixas vizinhas apenas encostadas tem sobreposicao pequena
    em relacao a area e NAO sao fundidas.
    """
    boxes = [dict(d) for d in dets]
    changed = True
    while changed:
        changed = False
        out: list[dict[str, Any]] = []
        while boxes:
            a = boxes.pop(0)
            i = 0
            while i < len(boxes):
                b = boxes[i]
                ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
                ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
                iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                smaller = min(a["w"] * a["h"], b["w"] * b["h"])
                if smaller > 0 and inter / smaller >= thresh:
                    x1, y1 = min(a["x"], b["x"]), min(a["y"], b["y"])
                    x2 = max(a["x"] + a["w"], b["x"] + b["w"])
                    y2 = max(a["y"] + a["h"], b["y"] + b["h"])
                    label = a["label"] if a["conf"] >= b["conf"] else b["label"]
                    a = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                         "label": label, "conf": max(a["conf"], b["conf"])}
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


class YoloDetector:
    """Detector de objetos por IA (Ultralytics YOLO).

    Ordem de preferencia do modelo:
    1. Modelo customizado treinado com os produtos do usuario (melhor precisao);
    2. Modelo open-vocabulary (YOLO-World) que detecta 'caixa/produto' por texto;
    3. Modelo COCO generico (fallback).
    """

    def __init__(
        self,
        conf: float = 0.40,
        iou: float = 0.35,        # NMS mais forte: funde deteccoes sobrepostas na mesma caixa
        min_conf: float = 0.55,   # so aceita deteccao confiante (corta falso positivo)
        max_aspect_ratio: float = 4.0,
        min_area_ratio: float = 0.0008,
        merge_overlap: float = 0.60,  # funde deteccoes da mesma caixa fisica
    ) -> None:
        self.merge_overlap = merge_overlap
        self.conf = conf              # confianca minima do YOLO no predict
        self.iou = iou               # NMS: funde deteccoes sobrepostas
        self.min_conf = min_conf     # descarte pos-predict de deteccoes fracas
        self.max_aspect_ratio = max_aspect_ratio  # caixa: proporcao sensata
        self.min_area_ratio = min_area_ratio      # ignora ruido minusculo
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

    # Classes que nunca sao caixa/produto (ignoradas mesmo se o modelo COCO detectar).
    _IGNORE_LABELS = {"person", "cat", "dog", "bird", "chair", "tvmonitor", "tv"}

    def detect(self, frame) -> list[dict[str, Any]]:
        """Retorna [{x,y,w,h,label,conf}] dos objetos detectados e filtrados.

        Filtros aplicados apos o predict:
        - descarta classes que nao sao objeto contavel (pessoa, animal...);
        - descarta confianca < min_conf;
        - descarta proporcao (aspect ratio) fora do esperado para caixa;
        - descarta area minuscula (ruido).
        """
        if self._model is None and not self.load():
            return []

        h_frame, w_frame = frame.shape[:2]
        min_area = self.min_area_ratio * (w_frame * h_frame)

        results = self._model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        out: list[dict[str, Any]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                label = self._names.get(cls, str(cls))
                if label in self._IGNORE_LABELS:
                    continue
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                if conf < self.min_conf:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                w, h = int(x2 - x1), int(y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                if (w * h) < min_area:
                    continue
                aspect = max(w, h) / min(w, h)
                if aspect > self.max_aspect_ratio:
                    continue
                out.append(
                    {"x": int(x1), "y": int(y1), "w": w, "h": h, "label": label, "conf": conf}
                )
        # Uma caixa 3D real pode disparar varias deteccoes (cada face). Funde.
        return _merge_overlapping(out, self.merge_overlap)

    def detect_in_roi(
        self, frame, roi_px: tuple[int, int, int, int]
    ) -> list[dict[str, Any]]:
        """Detecta apenas dentro da ROI (area vigiada), devolvendo coords no frame inteiro."""
        px1, py1, px2, py2 = roi_px
        if px2 - px1 < 8 or py2 - py1 < 8:
            return []
        crop = frame[py1:py2, px1:px2]
        dets = self.detect(crop)
        for d in dets:  # desloca de volta para o sistema de coordenadas do frame
            d["x"] += px1
            d["y"] += py1
        return dets
