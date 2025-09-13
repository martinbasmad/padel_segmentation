# modules/colorize_overlay.py
import cv2
import numpy as np
from typing import Optional

def overlay_by_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 255),   # BGR (amarillo)
    alpha: float = 0.6,                            # opacidad del color/colormap
    soften: int = 0,                               # radio de suavizado de bordes (px, 0 desactiva)
    colormap: Optional[int] = None                 # ej: cv2.COLORMAP_TURBO; None = color sólido
) -> np.ndarray:
    """
    Superpone color/colormap sobre 'frame_bgr' donde 'mask' es >0 (blanco).

    Parámetros:
      - frame_bgr: imagen original en BGR (uint8).
      - mask: máscara 0/255 (uint8); si viene en gris con otros valores, se umbraliza.
      - color: color BGR para modo sólido (si colormap=None).
      - alpha: opacidad del overlay en zona enmascarada [0..1].
      - soften: suaviza bordes de la máscara (GaussianBlur). Valores típicos: 3, 5, 7…
      - colormap: si no es None, aplica un colormap de OpenCV sobre la máscara (p.ej. cv2.COLORMAP_TURBO).

    Retorna:
      - frame resaltado (uint8), misma forma que frame_bgr.
    """
    assert frame_bgr.dtype == np.uint8, "frame_bgr debe ser uint8"
    h, w = frame_bgr.shape[:2]

    # Asegurar máscara binaria 0/255 (1 canal)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Normalizamos a 0/255
    _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Suavizado opcional de bordes
    if soften and soften > 0:
        # kernel impar
        k = max(1, int(soften))
        if k % 2 == 0:
            k += 1
        mask_bin = cv2.GaussianBlur(mask_bin, (k, k), 0)

    # Construir capa de color/colormap
    if colormap is None:
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:] = np.array(color, dtype=np.uint8)
    else:
        # Para colormap, usamos la máscara como intensidad (0-255)
        overlay = cv2.applyColorMap(mask_bin, colormap)

    # Mezcla sólo en la región de la máscara
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)

    # Convertir la máscara a 3 canales para hacer el “where”
    mask_3c = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    out = np.where(mask_3c > 0, blended, frame_bgr)

    return out
