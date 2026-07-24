from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .vision import require_cv2


def _most_common(values: list[int]) -> int:
    """Valor que mais aparece na janela (desempate: o maior). Suaviza tremor."""
    if not values:
        return 0
    counts = Counter(values)
    top = max(counts.values())
    return max(v for v, c in counts.items() if c == top)


def _overlaps_any(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    """True se `box` intersecta qualquer bbox em `others` (confirma que a mancha e caixa)."""
    x, y, w, h = box
    for (ox, oy, ow, oh) in others:
        ix = max(0, min(x + w, ox + ow) - max(x, ox))
        iy = max(0, min(y + h, oy + oh) - max(y, oy))
        if ix > 0 and iy > 0:
            return True
    return False


class ZoneMonitor:
    """Vigia uma area (ROI) e conta quantos objetos estao deixados/parados nela agora.

    Funciona por comparacao com o fundo vazio da area (camera fixa):
    - aprende como a area e quando esta vazia (calibracao);
    - a cada frame compara o atual com esse fundo;
    - ignora movimento (pessoa/mao passando) contando so quando a cena estabiliza;
    - reporta quantos objetos distintos estao presentes na area agora.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.015,
        diff_threshold: int = 30,
        motion_threshold: float = 3.5,
        stable_needed: int = 6,
        detector: Any | None = None,
    ) -> None:
        # detector: YoloDetector opcional. Quando presente e carregado, a contagem
        # usa reconhecimento real de objeto (caixa/produto) em vez de so blob de pixel.
        self.detector = detector
        self.detector_active = False  # vira True quando o YOLO carrega de fato
        self.roi: tuple[float, float, float, float] | None = None
        self.background = None  # grayscale borrado da area vazia
        self.prev_gray = None
        self.min_area_ratio = min_area_ratio
        self.diff_threshold = diff_threshold
        self.motion_threshold = motion_threshold
        self.stable_needed = stable_needed
        self.stable_frames = 0
        self.current_count = 0  # objetos visiveis agora (suavizado)
        self.total_count = 0    # contagem ATUAL da area (segue o current, nao acumula)
        self.baseline = 0
        self.boxes: list[tuple[int, int, int, int]] = []
        # Suavizacao: contagem reportada = valor mais frequente das ultimas leituras
        # estaveis. Impede que um frame ruim do detector faca o numero saltar.
        self._smooth_window = 7
        self._recent_visible: list[int] = []
        self._last_mask = None      # mascara de diferenca do ultimo frame (ROI-local)
        self._last_roi_bgr = None   # recorte BGR da area no ultimo frame (p/ coleta)

    @property
    def active(self) -> bool:
        return self.roi is not None

    def set_roi(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.background = None  # forca recalibrar com a area recem-definida
        self.prev_gray = None
        self.stable_frames = 0
        self.current_count = 0
        self.total_count = 0
        self.baseline = 0
        self.boxes = []
        self._recent_visible = []

    def clear(self) -> None:
        self.roi = None
        self.background = None
        self.prev_gray = None
        self.current_count = 0
        self.total_count = 0
        self.baseline = 0
        self.boxes = []
        self._recent_visible = []
        self._last_mask = None
        self._last_roi_bgr = None

    def reset_total(self) -> None:
        self.total_count = 0
        self.current_count = 0
        self._recent_visible = []

    def recalibrate(self) -> None:
        """Marca para capturar novo fundo vazio no proximo frame."""
        self.background = None
        self.stable_frames = 0
        self.total_count = 0
        self.baseline = 0
        self._recent_visible = []

    def _roi_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.roi  # type: ignore[misc]
        px1 = max(0, int(x1 * width))
        py1 = max(0, int(y1 * height))
        px2 = min(width, int(x2 * width))
        py2 = min(height, int(y2 * height))
        return px1, py1, px2, py2

    def _segment_objects(self, roi_bgr, mask, min_area: float, ox: int, oy: int):
        """Separa objetos, inclusive encostados, usando watershed."""
        cv2 = require_cv2()
        if cv2.countNonZero(mask) < min_area:
            return []

        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)

        # Distancia até a borda: os "centros" dos objetos ficam com valor alto.
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return []
        _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num, markers = cv2.connectedComponents(sure_fg)
        if num <= 1:
            return []
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(roi_bgr, markers)

        boxes: list[tuple[int, int, int, int]] = []
        for label in range(2, markers.max() + 1):
            ys, xs = np.where(markers == label)
            if xs.size < min_area:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            boxes.append((ox + x1, oy + y1, x2 - x1, y2 - y1))
        return boxes

    def process(self, frame) -> dict[str, Any] | None:
        cv2 = require_cv2()
        if self.roi is None:
            return None

        height, width = frame.shape[:2]
        px1, py1, px2, py2 = self._roi_pixels(width, height)
        if px2 - px1 < 5 or py2 - py1 < 5:
            return None

        roi = frame[py1:py2, px1:px2]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # Primeiro frame apos definir a area: aprende o fundo vazio.
        if self.background is None or self.background.shape != gray.shape:
            self.background = gray
            self.prev_gray = gray
            self.current_count = 0
            self.boxes = []
            return {
                "count": 0,
                "boxes": [],
                "roi_px": (px1, py1, px2, py2),
                "stable": False,
                "calibrating": True,
                "detector": "yolo" if self.detector_active else "movimento",
            }

        # Detecta instabilidade (movimento) comparando frame atual com o anterior.
        motion = 0.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            motion = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
        self.prev_gray = gray

        if motion > self.motion_threshold:
            self.stable_frames = 0
        else:
            self.stable_frames += 1

        # Diferenca com o fundo vazio -> o que foi deixado na area.
        delta = cv2.absdiff(self.background, gray)
        _, thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        area_total = (px2 - px1) * (py2 - py1)
        min_area = self.min_area_ratio * area_total
        roi_bgr = frame[py1:py2, px1:px2]
        self._last_mask = mask
        self._last_roi_bgr = roi_bgr
        boxes = self._segment_objects(roi_bgr, mask, min_area, px1, py1)

        stable = self.stable_frames >= self.stable_needed
        # So conta quando a cena esta estavel (sem gente/mao mexendo).
        if stable:
            # ESTRATEGIA (camera fixa + fundo vazio calibrado):
            # a CONTAGEM vem das manchas fisicas (o que foi deixado na area) —
            # 1 objeto solido = 1 mancha conectada, robusto a caixa 3D.
            # O YOLO so CONFIRMA que a mancha e caixa (rejeita pessoa/sombra);
            # se o YOLO nao detectar nada (possivel falha), confia nas manchas.
            blobs_local = self._merged_blobs(mask)
            blobs = [(x + px1, y + py1, w, h) for (x, y, w, h) in blobs_local]

            if self.detector is not None:
                yolo_boxes = self._detect_with_yolo(frame, (px1, py1, px2, py2)) or []
                if yolo_boxes:
                    confirmed = [b for b in blobs if _overlaps_any(b, yolo_boxes)]
                    final_boxes = confirmed if confirmed else blobs
                else:
                    final_boxes = blobs
            else:
                final_boxes = blobs

            # Suaviza: reporta a contagem mais frequente das ultimas leituras estaveis.
            self._recent_visible.append(len(final_boxes))
            if len(self._recent_visible) > self._smooth_window:
                self._recent_visible.pop(0)
            stabilized = _most_common(self._recent_visible)

            self.current_count = stabilized
            self.total_count = stabilized  # contagem ATUAL (nao acumula tremor)
            self.boxes = final_boxes

        return {
            "count": self.total_count,
            "visible": self.current_count,
            "boxes": self.boxes,
            "roi_px": (px1, py1, px2, py2),
            "stable": stable,
            "calibrating": False,
            "detector": "yolo" if self.detector_active else "movimento",
        }

    def _merged_blobs(self, mask) -> list[tuple[int, int, int, int]]:
        """Manchas fisicas na area (ROI-local), fundindo fragmentos da mesma caixa.

        Dilata a mascara de diferenca para juntar pedacos (fita, costura, sombra
        colada) num unico bloco, depois aperta o bbox de volta na mascara real.
        """
        cv2 = require_cv2()
        big = cv2.dilate(mask, np.ones((25, 25), np.uint8))
        num, _labels, stats, _ = cv2.connectedComponentsWithStats(big)
        total_px = mask.shape[0] * mask.shape[1]
        out: list[tuple[int, int, int, int]] = []
        for i in range(1, num):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < 0.005 * total_px:
                continue
            sub = mask[y:y + h, x:x + w]
            ys, xs = np.nonzero(sub)
            if xs.size == 0:
                continue
            tx1, ty1 = x + int(xs.min()), y + int(ys.min())
            tx2, ty2 = x + int(xs.max()), y + int(ys.max())
            if tx2 - tx1 < 12 or ty2 - ty1 < 12:
                continue
            out.append((tx1, ty1, tx2 - tx1, ty2 - ty1))
        return out

    def label_blobs(self) -> list[tuple[int, int, int, int]] | None:
        """Blocos fisicos (ROI-local) para rotulo automatico na coleta de dataset."""
        if self._last_mask is None:
            return None
        return self._merged_blobs(self._last_mask)

    def _detect_with_yolo(
        self, frame, roi_px: tuple[int, int, int, int]
    ) -> list[tuple[int, int, int, int]] | None:
        """Roda YOLO na ROI. Retorna caixas (x,y,w,h) ou None se indisponivel/erro."""
        if self.detector is None:
            return None
        try:
            if not self.detector_active:
                if not self.detector.load():
                    self.detector = None  # falhou: nao tenta de novo
                    return None
                self.detector_active = True
            dets = self.detector.detect_in_roi(frame, roi_px)
            return [(d["x"], d["y"], d["w"], d["h"]) for d in dets]
        except Exception:
            return None  # qualquer falha: cai no fallback de movimento


def draw_zone_overlay(frame, result: dict[str, Any]):
    """Desenha a area, os objetos detectados e o contador sobre o frame."""
    cv2 = require_cv2()
    px1, py1, px2, py2 = result["roi_px"]
    stable = result.get("stable", False)
    calibrating = result.get("calibrating", False)

    # Retangulo da area vigiada.
    zone_color = (13, 200, 150) if stable else (60, 160, 255)
    cv2.rectangle(frame, (px1, py1), (px2, py2), zone_color, 3)

    # Objetos presentes.
    for (x, y, w, h) in result.get("boxes", []):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 2)

    # Cabecalho com a contagem.
    if calibrating:
        label = "Aprendendo a area vazia..."
    elif not stable:
        label = "Movimento... aguardando estabilizar"
    else:
        label = f"Total: {result['count']}  (visiveis: {result.get('visible', 0)})"

    cv2.rectangle(frame, (px1, max(0, py1 - 40)), (px1 + 460, py1), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (px1 + 8, max(20, py1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return frame
