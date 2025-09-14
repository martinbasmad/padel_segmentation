# modules/filter_roundness.py
import cv2
import numpy as np
from typing import Optional

def filter_by_roundness(
    mask: np.ndarray,
    min_circularity: Optional[float] = None,
    max_circularity: Optional[float] = None,
) -> np.ndarray:
    """
    Filtra los componentes de la máscara por 'circularity' = 4*pi*area / perimetro^2 (∈ [0,1]).
    1.0 ≈ círculo perfecto. Valores típicos para objetos redondeados: >= 0.7.

    Si min_circularity es None, no aplica filtro inferior.
    Si max_circularity es None, no aplica filtro superior.
    """
    if min_circularity is None and max_circularity is None:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue

        circularity = float(4.0 * np.pi * area / (per * per))

        if (min_circularity is not None and circularity < min_circularity):
            continue
        if (max_circularity is not None and circularity > max_circularity):
            continue

        cv2.drawContours(out, [cnt], -1, 255, thickness=cv2.FILLED)

    return out
