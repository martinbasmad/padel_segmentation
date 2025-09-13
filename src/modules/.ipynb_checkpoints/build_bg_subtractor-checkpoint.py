import cv2
from typing import Literal

def build_bg_subtractor(
    algo: Literal["mog2", "knn"] = "mog2",
    history: int = 500,
    var_threshold: float = 16.0,
    detect_shadows: bool = True
):
    algo = str(algo).strip().lower()
    if algo in ("mog2", "mog", "gmog2"):
        return cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=bool(detect_shadows),
        )
    elif algo == "knn":
        return cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=var_threshold,
            detectShadows=bool(detect_shadows),
        )
    else:
        raise ValueError(f"Algoritmo no soportado: {algo}. Use 'mog2' o 'knn'.")