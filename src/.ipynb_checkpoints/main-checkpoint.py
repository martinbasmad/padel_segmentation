# src/main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import modules.process_video as pv
from modules.substract_artificial_background import save_median_background
from modules.process_by_threshold import process_video_by_threshold
import os


def main():
    # Archivo de entrada (relativo a src/ → ../data/...)
    filepath = os.path.join("..", "data", "padel_amateur.mp4")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encuentra el archivo de entrada: {filepath}")

    # Carpeta donde está el video de entrada
    data_dir = os.path.dirname(filepath)

    # Nombre base y extensión
    name, ext = os.path.splitext(os.path.basename(filepath))

    # Carpeta de resultados y archivos de salida
    results_folder = os.path.join(data_dir, "results")
    mask_path = os.path.join(results_folder, f"{name}_mask{ext}")
    overlay_path = os.path.join(results_folder, f"{name}_overlay{ext}")
    bg_png_path = os.path.join(results_folder, f"{name}_background.png")
    otsu_mask_path = os.path.join(results_folder, f"{name}_mask_otsu{ext}")
    otsu_overlay_path = os.path.join(results_folder, f"{name}_overlay_otsu{ext}")

    # Crear carpetas si no existen
    os.makedirs(results_folder, exist_ok=True)

    # === 0) Cálculo y guardado del fondo artificial por mediana ===
    save_median_background(
        input_path=filepath,
        output_png_path=bg_png_path,
        sample_size=100,   # modificar según se desee
        seed=42
    )

    # Parámetros globales de filtrado por área
    pv.MIN_SIZE = 0
    pv.MAX_SIZE = 0

    # === 1) Pipeline con sustracción de fondo (MOG2/KNN) ===
    common = dict(
        input_path=filepath,
        algo="mog2",          # o "knn"
        history=2000,
        varth=500.0,
        shadows=False,
        thresh=250,
        kernel=3,             # 1 = desactiva morfología
        fade=0.3,
        bin_level=32,
        min_circularity=None,  # None para desactivar
        max_circularity=None,
    )

    # 1a) Generar video de MÁSCARA (blanco/negro)
    pv.process_video(
        output_path=mask_path,
        **common
    )

    # 1b) Generar video COLOREADO (overlay sobre el frame original)
    pv.process_video(
        output_path=overlay_path,
        write_overlay=True,
        overlay_color=(0, 0, 255),   # BGR: rojo
        overlay_alpha=0.6,
        overlay_soften=3,
        overlay_colormap=None,       # para heatmap: cv2.COLORMAP_TURBO
        **common
    )

    # === 2) Pipeline por resta de background artificial + Otsu ===
    # 2a) Máscara Otsu
    process_video_by_threshold(
        input_path=filepath,
        background_image_path=bg_png_path,
        output_path=otsu_mask_path,
        morph_kernel=3,
        blur_ksize=3,
        write_overlay=False,
    )

    # 2b) Overlay Otsu
    process_video_by_threshold(
        input_path=filepath,
        background_image_path=bg_png_path,
        output_path=otsu_overlay_path,
        morph_kernel=3,
        blur_ksize=3,
        write_overlay=True,
        overlay_color=(0, 255, 0),  # ejemplo: verde
        overlay_alpha=0.6,
        overlay_soften=3,
        overlay_colormap=None,
    )


if __name__ == "__main__":
    main()
