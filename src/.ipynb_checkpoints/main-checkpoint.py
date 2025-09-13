#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import modules.process_video as pv
import os


def main():
    # Archivo de entrada
    filepath = os.path.join("..", "data", "padel_amateur.mp4")

    # Paths
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encuentra el archivo de entrada: {filepath}")

    name, ext = os.path.splitext(os.path.basename(filepath))
    mask_path = os.path.join("data", f"{name}_mask{ext}")
    overlay_path = os.path.join("data", f"{name}_overlay{ext}")

    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)

    # Parámetros globales de filtrado por área (desactivados)
    pv.MIN_SIZE = 0
    pv.MAX_SIZE = 0

    # Parámetros comunes
    common = dict(
        input_path=filepath,
        algo="knn",     # o "mog2"
        history=50,
        varth=16.0,
        shadows=False,
        thresh=15,
        kernel=3,       # 1 = desactiva morfología
        fade=0.10,
        bin_level=32,
    )

    # 1) Generar video de MÁSCARA (blanco/negro)
    pv.process_video(
        output_path=mask_path,
        **common
    )

    # 2) Generar video COLOREADO (overlay sobre el frame original)
    pv.process_video(
        output_path=overlay_path,
        write_overlay=True,
        overlay_color=(0, 0, 255),   # BGR: rojo
        overlay_alpha=0.6,
        overlay_soften=3,
        overlay_colormap=None,       # usar color sólido; para heatmap: cv2.COLORMAP_TURBO
        **common
    )


if __name__ == "__main__":
    main()
