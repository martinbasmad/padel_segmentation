import cv2
import numpy as np

def apply_morph(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask