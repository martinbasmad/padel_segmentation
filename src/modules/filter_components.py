# modules/remove_small_components.py
import cv2          # connectedComponentsWithStats, CC_STAT_AREA
import numpy as np  # zeros_like

def filter_components(mask: np.ndarray, min_size: int = 0, max_size: int = 0) -> np.ndarray:
    """
    Filtra componentes conectados en una máscara binaria según área mínima y/o máxima.
    - mask: binaria 0/255 (uint8)
    - min_size: área mínima (<=1 desactiva)
    - max_size: área máxima (<=0 desactiva)
    """
    if (min_size <= 1) and (max_size <= 0):
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):  # 0 = fondo
        area = stats[i, cv2.CC_STAT_AREA]
        if (min_size <= 1 or area >= min_size) and (max_size <= 0 or area <= max_size):
            cleaned[labels == i] = 255

    return cleaned
