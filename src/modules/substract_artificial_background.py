# modules/substract_artificial_background.py
import cv2
import numpy as np
import random

def compute_median_background(input_path: str, sample_size: int = 100, seed: int = 42) -> np.ndarray:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("El video no contiene frames válidos.")

    k = min(sample_size, total_frames)
    rng = random.Random(seed)
    # Muestras al azar sin reemplazo
    indices = sorted(rng.sample(range(total_frames), k))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No se pudieron leer frames para calcular la mediana.")

    # Mediana por píxel y canal
    stack = np.stack(frames, axis=0).astype(np.float32)
    median = np.median(stack, axis=0)
    bg = np.clip(median, 0, 255).astype(np.uint8)
    return bg

def save_median_background(input_path: str, output_png_path: str, sample_size: int = 100, seed: int = 42) -> None:
    bg = compute_median_background(input_path=input_path, sample_size=sample_size, seed=seed)
    # Guardar en BGR (cv2.imwrite espera BGR)
    ok = cv2.imwrite(output_png_path, bg)
    if not ok:
        raise RuntimeError(f"No se pudo guardar la imagen en: {output_png_path}")
