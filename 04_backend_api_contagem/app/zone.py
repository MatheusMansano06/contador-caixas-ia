from __future__ import annotations

from typing import Any

import numpy as np

from .vision import require_cv2


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
    ) -> None:
        self.roi: tuple[float, float, float, float] | None = None
        self.background = None  # grayscale borrado da area vazia
        self.prev_gray = None
        self.min_area_ratio = min_area_ratio
        self.diff_threshold = diff_threshold
        self.motion_threshold = motion_threshold
        self.stable_needed = stable_needed
        self.stable_frames = 0
        self.current_count = 0  # objetos visiveis agora
        self.total_count = 0    # acumulado: cada caixa colocada soma +1, nunca desconta
        self.baseline = 0       # objetos visiveis na ultima cena estavel
        self.boxes: list[tuple[int, int, int, int]] = []

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

    def clear(self) -> None:
        self.roi = None
        self.background = None
        self.prev_gray = None
        self.current_count = 0
        self.total_count = 0
        self.baseline = 0
        self.boxes = []

    def reset_total(self) -> None:
        self.total_count = 0
        self.baseline = self.current_count

    def recalibrate(self) -> None:
        """Marca para capturar novo fundo vazio no proximo frame."""
        self.background = None
        self.stable_frames = 0
        self.total_count = 0
        self.baseline = 0

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
        boxes = self._segment_objects(roi_bgr, mask, min_area, px1, py1)

        stable = self.stable_frames >= self.stable_needed
        # So confirma a contagem quando a cena esta estavel (sem gente/mao mexendo).
        if stable:
            visible = len(boxes)
            # Cada caixa nova colocada soma ao total; empilhar/tapar nao desconta.
            if visible > self.baseline:
                self.total_count += visible - self.baseline
            self.baseline = visible
            self.current_count = visible
            self.boxes = boxes

        return {
            "count": self.total_count,
            "visible": self.current_count,
            "boxes": self.boxes,
            "roi_px": (px1, py1, px2, py2),
            "stable": stable,
            "calibrating": False,
        }


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
