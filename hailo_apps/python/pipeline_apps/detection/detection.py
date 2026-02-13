# region imports
# Standard library imports
import argparse

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
def _parse_line_norm(s: str):
    # Accept "x1,y1,x2,y2" normalized to [0..1]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--line must be 'x1,y1,x2,y2' (normalized 0..1)")
    vals = tuple(float(v) for v in parts)
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise argparse.ArgumentTypeError("--line values must be in [0..1]")
    if vals[0] == vals[2] and vals[1] == vals[3]:
        raise argparse.ArgumentTypeError("--line points must not be identical")
    return vals


def _line_points_px(line_norm, width, height):
    x1n, y1n, x2n, y2n = line_norm
    return (int(x1n * width), int(y1n * height)), (int(x2n * width), int(y2n * height))


def _point_side_of_line(p1, p2, q):
    """Signed side test for directed line p1->p2. >0 left, <0 right, 0 on line."""
    x1, y1 = p1
    x2, y2 = p2
    x, y = q
    dx = x2 - x1
    dy = y2 - y1
    return dx * (y - y1) - dy * (x - x1)


def _bbox_intersects_infinite_line(p1, p2, xyxy):
    """Return True if bbox corners lie on different sides of the line (or on the line)."""
    x1, y1, x2, y2 = xyxy
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    has_pos = False
    has_neg = False
    for c in corners:
        s = _point_side_of_line(p1, p2, c)
        if s > 0:
            has_pos = True
        elif s < 0:
            has_neg = True
        else:
            return True
    return has_pos and has_neg


class user_app_callback_class(app_callback_class):
    def __init__(self, line_norm, red_side, red_if_crossing):
        super().__init__()
        self.new_variable = 42

        # Defaults (can be overridden by CLI args parsed in main)
        self.line_norm = line_norm  # (x1,y1,x2,y2) normalized to [0..1]
        self.red_side = red_side  # 'left' or 'right' relative to directed line (p1->p2)
        self.red_if_crossing = red_if_crossing

    def new_function(self):
        return "The meaning of life is: "


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def _print_red(msg: str):
    # ANSI 24-bit red
    print(f"\x1b[38;2;255;0;0m{msg}\x1b[0m")


def _get_detection_bbox_xyxy(detection, width, height):
    """Return (x1,y1,x2,y2) in pixels if bbox is available, else None."""
    if width is None or height is None:
        return None

    bbox_fn = getattr(detection, "get_bbox", None)
    if not callable(bbox_fn):
        return None

    b = bbox_fn()

    # Prefer HailoBBox method API (normalized)
    if all(callable(getattr(b, name, None)) for name in ("xmin", "ymin", "xmax", "ymax")):
        x1 = float(b.xmin())
        y1 = float(b.ymin())
        x2 = float(b.xmax())
        y2 = float(b.ymax())
        return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

    # Fallback: properties
    if all(hasattr(b, name) for name in ("xmin", "ymin", "xmax", "ymax")):
        x1 = float(b.xmin)
        y1 = float(b.ymin)
        x2 = float(b.xmax)
        y2 = float(b.ymax)
        return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

    return None


def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    # Note: Frame counting is handled automatically by the framework wrapper
    frame_idx = user_data.get_count()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0

    p1 = p2 = None
    if width is not None and height is not None:
        try:
            p1, p2 = _line_points_px(user_data.line_norm, width, height)
        except Exception:
            p1 = p2 = None

    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()

        xyxy = None
        is_crossing = False
        is_red = False
        if p1 is not None and p2 is not None and height is not None:
            xyxy = _get_detection_bbox_xyxy(detection, width, height)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                side = _point_side_of_line(p1, p2, (cx, cy))
                if user_data.red_side == "left":
                    is_red = side > 0
                else:
                    is_red = side < 0

                is_crossing = _bbox_intersects_infinite_line(p1, p2, xyxy)
                if user_data.red_if_crossing and is_crossing:
                    is_red = True

        # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        if is_crossing:
            _print_red(
                f"CROSSED line | ID={track_id} | Label={label} | Conf={confidence:.2f}"
            )

        # Draw bbox (frame is RGB here)
        if frame is not None and xyxy is not None:
            x1, y1, x2, y2 = xyxy
            color = (255, 0, 0) if is_red else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label + track ID above the box
            safe_label = label if label is not None else ""
            label_text = f"{safe_label} ID:{track_id}".strip()
            text_scale = 0.6
            text_thickness = 2
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            tx = max(0, x1)
            ty = max(th + 2, y1 - 4)
            cv2.rectangle(
                frame,
                (tx, ty - th - baseline - 2),
                (tx + tw + 6, ty + baseline + 2),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label_text,
                (tx + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA,
            )

        if label == "person":
            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            )
            detection_count += 1
    if user_data.use_frame:
        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{user_data.new_function()} {user_data.new_variable}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw the user-defined angled line
        if frame is not None and p1 is not None and p2 is not None:
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return


def main():
    hailo_logger.info("Starting Detection App.")

    parser = get_pipeline_parser()
    parser.add_argument(
        "--line",
        type=_parse_line_norm,
        default=(0.5, 0.1, 0.5, 0.9),
        help="Directed line defined by normalized coordinates x1,y1,x2,y2 (0..1) for red detection",
    )
    parser.add_argument(
        "--red-side",
        type=str,
        choices=["left", "right"],
        default="right",
        help="Which side of the line to mark as red (relative to directed line p1->p2)",
    )
    parser.add_argument(
        "--red-if-crossing",
        action="store_true",
        default=False,
        help="Mark as red if bbox crosses the line, regardless of which side the center is on",
    )
    user_data = user_app_callback_class(
        line_norm=parser.parse_args().line, red_side=parser.parse_args().red_side, red_if_crossing=parser.parse_args().red_if_crossing)
    app = GStreamerDetectionApp(app_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()

#running example
#python3  hailo_apps.python.pipeline_apps.detection.detection --line 0.1,0.1,0.9,0.9 --red-side right --red-if-crossing`
