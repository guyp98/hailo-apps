# region imports
# Standard library imports

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

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42

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
    line_x = int(0.5 * width) if width is not None else None
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()

        xyxy = None
        is_right = False
        is_crossing = False
        if line_x is not None and height is not None:
            xyxy = _get_detection_bbox_xyxy(detection, width, height)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                is_crossing = x1 <= line_x <= x2
                # Entire bbox is to the right of the line
                is_right = x1 >= line_x

        # Red only if fully to the right
        is_red = is_right

        # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        if is_crossing:
            _print_red(
                f"CROSSED line_x={line_x} | ID={track_id} | Label={label} | Conf={confidence:.2f}"
            )

        # Draw bbox (frame is RGB here)
        if frame is not None and xyxy is not None:
            x1, y1, x2, y2 = xyxy
            color = (255, 0, 0) if is_red else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw track ID label above the box
            label_text = f"ID:{track_id}"
            text_scale = 0.6
            text_thickness = 2
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            tx = max(0, x1)
            ty = max(th + 2, y1 - 4)
            # background for readability
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

        # Draw a vertical reference line overlay (frame is RGB here)
        if frame is not None and line_x is not None and height is not None:
            p1 = (line_x, int(0.1 * height))
            p2 = (line_x, int(0.9 * height))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return


def main():
    hailo_logger.info("Starting Detection App.")
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
