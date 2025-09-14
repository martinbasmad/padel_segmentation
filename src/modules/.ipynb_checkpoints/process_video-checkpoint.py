# modules/process_video.py
import cv2
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Tuple
from tqdm import tqdm

from .build_bg_subtractor import build_bg_subtractor
from .apply_morph import apply_morph
from .filter_components import filter_components
from .colorize_overlay import overlay_by_mask
from .filter_roundness import filter_by_roundness  # ya creado por vos

# Variables globales (las setea main.py)
MIN_SIZE = 0   # <=1 desactiva mínimo
MAX_SIZE = 0   # <=0 desactiva máximo


def process_video(
    input_path: str,
    output_path: str,
    algo: Literal["mog2","knn"] = "mog2",
    history: int = 500,
    varth: float = 16.0,
    shadows: bool = False,
    thresh: int = 25,
    kernel: int = 3,
    fade: float = 0.90,
    bin_level: int = 32,
    # —— Opcional: salida con overlay coloreado en vez de máscara B/N ——
    write_overlay: bool = False,
    overlay_color: Tuple[int, int, int] = (0, 0, 255),  # BGR (rojo)
    overlay_alpha: float = 0.6,
    overlay_soften: int = 3,
    overlay_colormap: Optional[int] = None,  # p.ej. cv2.COLORMAP_TURBO
    # —— Filtrado por redondez/circularidad ——
    min_circularity: Optional[float] = None,  # e.g. 0.7
    max_circularity: Optional[float] = None,  # e.g. 1.0
):
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de entrada: {in_path}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video de entrada.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps if fps and fps > 0 else 30.0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # para tqdm

    if width <= 0 or height <= 0:
        ok, tmp = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("No se pudo leer el primer frame.")
        height, width = tmp.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ksize = max(1, int(kernel))
    if ksize % 2 == 0:
        ksize += 1
    fade = float(np.clip(fade, 0.0, 1.0))
    thresh = max(0, int(thresh))
    bin_level = int(np.clip(bin_level, 0, 255))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

    sub = build_bg_subtractor(
        algo=algo, history=history, var_threshold=varth, detect_shadows=bool(shadows)
    )
    trail = np.zeros((height, width), dtype=np.float32)

    # Progreso
    with tqdm(total=total_frames if total_frames > 0 else None,
              desc="Procesando video",
              unit="frame") as pbar:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            fg = sub.apply(frame, learningRate=0.005)

            if thresh > 0:
                _, fg = cv2.threshold(fg, thresh, 255, cv2.THRESH_BINARY)

            # Morfología opcional: kernel=1 => desactivada
            if ksize > 1:
                fg = apply_morph(fg, ksize)

            fg_norm = fg.astype(np.float32) / 255.0

            # Estela con desvanecimiento
            trail = trail * fade + fg_norm * (1.0 - fade)

            # Umbral final para binarizar la estela acumulada
            trail_8u = np.uint8(np.clip(trail * 255.0, 0, 255))
            _, mask_bin = cv2.threshold(trail_8u, bin_level, 255, cv2.THRESH_BINARY)

            # Filtrado por área mínima y/o máxima (si está activado)
            if (MIN_SIZE and MIN_SIZE > 1) or (MAX_SIZE and MAX_SIZE > 0):
                mask_bin = filter_components(
                    mask_bin, min_size=int(MIN_SIZE), max_size=int(MAX_SIZE)
                )

            # Filtrado por circularidad (roundness)
            if (min_circularity is not None) or (max_circularity is not None):
                mask_bin = filter_by_roundness(
                    mask=mask_bin,
                    min_circularity=min_circularity,
                    max_circularity=max_circularity,
                )

            if write_overlay:
                # Escribir frame original coloreado según máscara
                colored = overlay_by_mask(
                    frame_bgr=frame,
                    mask=mask_bin,
                    color=overlay_color,
                    alpha=overlay_alpha,
                    soften=overlay_soften,
                    colormap=overlay_colormap,
                )
                writer.write(colored)
            else:
                # Escribir máscara en B/N
                mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
                writer.write(mask_bgr)

            pbar.update(1)

    cap.release()
    writer.release()
