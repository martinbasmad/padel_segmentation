# modules/process_by_threshold.py
import cv2
import numpy as np
from tqdm import tqdm
from .colorize_overlay import overlay_by_mask

def process_video_by_threshold(
    input_path: str,
    background_image_path: str,
    output_path: str,
    morph_kernel: int = 3,
    blur_ksize: int = 3,
    # —— Overlay, igual que en process_video ——
    write_overlay: bool = False,
    overlay_color: tuple[int, int, int] = (0, 0, 255),  # BGR
    overlay_alpha: float = 0.6,
    overlay_soften: int = 3,
    overlay_colormap: int | None = None,  # p.ej., cv2.COLORMAP_TURBO
):
    """
    Resta un background artificial (imagen) a cada frame del video y aplica Otsu
    para obtener una máscara binaria. Si write_overlay=True, guarda overlay
    coloreado; si no, guarda la máscara B/N como video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    # Propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        ok, tmp = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("No se pudo leer el primer frame.")
        height, width = tmp.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Cargar background
    bg = cv2.imread(background_image_path, cv2.IMREAD_COLOR)
    if bg is None:
        cap.release()
        raise RuntimeError(f"No se pudo cargar el background: {background_image_path}")

    bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_AREA)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    # Normalizar parámetros
    mk = max(1, int(morph_kernel))
    if mk % 2 == 0:
        mk += 1
    bk = max(1, int(blur_ksize))
    if bk % 2 == 0:
        bk += 1

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height), True)

    kernel = None if mk <= 1 else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))

    with tqdm(total=total_frames if total_frames > 0 else None,
              desc="Procesando (bg-sub + Otsu)",
              unit="frame") as pbar:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resta absoluta con el fondo
            diff = cv2.absdiff(gray, bg_gray)

            # Suavizado opcional
            if bk > 1:
                diff = cv2.GaussianBlur(diff, (bk, bk), 0)

            # Otsu
            _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Morfología opcional
            if kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            if write_overlay:
                # Guardar overlay coloreado como en process_video
                colored = overlay_by_mask(
                    frame_bgr=frame,
                    mask=mask,
                    color=overlay_color,
                    alpha=overlay_alpha,
                    soften=overlay_soften,
                    colormap=overlay_colormap,
                )
                writer.write(colored)
            else:
                # Guardar máscara B/N
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                writer.write(mask_bgr)

            pbar.update(1)

    cap.release()
    writer.release()
