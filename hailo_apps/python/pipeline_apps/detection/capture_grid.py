"""Capture a single camera frame and overlay a normalized-coordinate grid.

Saves the annotated image so you can view it on another machine (e.g. via
scp) and visually pick polygon zone coordinates for the detection YAML config.

Usage:
    hailo-capture-grid --input usb
    hailo-capture-grid --input /dev/video8 --output /tmp/grid.jpg
    hailo-capture-grid --input rpi --resolution hd
"""

import argparse
import sys

import cv2
import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.toolbox import init_input_source

logger = get_logger(__name__)

GRID_STEP = 0.1  # line every 10%
LABEL_STEP = 0.2  # text label every 20%


def _draw_grid(frame: np.ndarray) -> np.ndarray:
    """Draw a normalized coordinate grid on *frame* (BGR, modified in-place)."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay for the grid lines
    overlay = frame.copy()

    # --- grid lines every 0.1 ---
    for i in range(1, int(1.0 / GRID_STEP)):
        norm = round(i * GRID_STEP, 2)
        px_x = int(norm * w)
        px_y = int(norm * h)

        # Thicker + brighter for the center lines (0.5)
        if abs(norm - 0.5) < 1e-6:
            color, thickness = (0, 255, 255), 2  # cyan
        else:
            color, thickness = (200, 200, 200), 1  # light gray

        cv2.line(overlay, (px_x, 0), (px_x, h), color, thickness)
        cv2.line(overlay, (0, px_y), (w, px_y), color, thickness)

    # Blend overlay onto frame (alpha)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # --- coordinate labels every 0.2 ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(w, h) / 2000)
    thick = 1

    for ix in range(0, int(1.0 / LABEL_STEP) + 1):
        nx = round(ix * LABEL_STEP, 1)
        px_x = int(nx * w)
        for iy in range(0, int(1.0 / LABEL_STEP) + 1):
            ny = round(iy * LABEL_STEP, 1)
            px_y = int(ny * h)

            label = f"{nx:.1f},{ny:.1f}"
            (tw, th_txt), _ = cv2.getTextSize(label, font, scale, thick)

            # Offset so text doesn't sit on the edge
            tx = min(px_x + 3, w - tw - 2)
            ty = min(px_y + th_txt + 3, h - 2)

            # Dark background rectangle for readability
            cv2.rectangle(frame, (tx - 1, ty - th_txt - 2), (tx + tw + 2, ty + 3), (0, 0, 0), -1)
            cv2.putText(frame, label, (tx, ty), font, scale, (0, 255, 255), thick, cv2.LINE_AA)

    # Border labels: axes explanation
    cv2.putText(frame, "x ->", (w // 2 - 20, 15), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "y v", (3, h // 2), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Capture a camera frame with a normalized coordinate grid overlay.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="usb",
        help="Input source: 'usb', 'rpi', '/dev/videoX', or a video/image file path (default: usb)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="zone_grid.jpg",
        help="Output image path (default: zone_grid.jpg)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Camera resolution: sd (640x480), hd (1280x720), fhd (1920x1080). Default: camera native.",
    )
    args = parser.parse_args()

    # --- grab a single frame ---
    cap, images = init_input_source(args.input, batch_size=1, resolution=args.resolution)

    frame = None
    if cap is not None:
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            logger.error("Failed to read a frame from the camera.")
            sys.exit(1)
    elif images:
        frame = images[0]
        # images from init_input_source are RGB; convert to BGR for cv2 ops
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        logger.error("No frame could be captured from '%s'.", args.input)
        sys.exit(1)

    logger.info("Captured frame: %dx%d", frame.shape[1], frame.shape[0])

    # --- draw grid and save ---
    _draw_grid(frame)
    cv2.imwrite(args.output, frame)
    print(f"Grid image saved to: {args.output}")


if __name__ == "__main__":
    main()
