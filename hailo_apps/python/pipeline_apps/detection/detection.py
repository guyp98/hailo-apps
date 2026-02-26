# region imports
# Standard library imports
import argparse
import time

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2
import numpy as np
import yaml

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
def _parse_zone_norm(s: str):
    """Parse 'x1,y1,x2,y2,x3,y3,...' into list of (x,y) tuples, normalized to [0..1]."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise argparse.ArgumentTypeError(
            "--zone must be 'x1,y1,x2,y2,x3,y3,...' with at least 3 vertices (6 values)"
        )
    vals = [float(v) for v in parts]
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise argparse.ArgumentTypeError("--zone values must be in [0..1]")
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


def _load_detection_config(config_path):
    """Load detection-specific settings from a YAML file.

    Returns dict with keys: zone, dwell_threshold, filter_labels.
    Missing keys are not included (caller uses CLI defaults).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    result = {}
    if "zone" in cfg and cfg["zone"] is not None:
        result["zone"] = [(float(p[0]), float(p[1])) for p in cfg["zone"]]
    if "dwell_threshold" in cfg:
        result["dwell_threshold"] = float(cfg["dwell_threshold"])
    if "filter_labels" in cfg and cfg["filter_labels"] is not None:
        result["filter_labels"] = [str(l) for l in cfg["filter_labels"]]
    return result


def _zone_points_px(zone_norm, width, height):
    """Convert normalized zone vertices to pixel coordinates."""
    return [(int(x * width), int(y * height)) for x, y in zone_norm]


def _point_in_polygon(polygon, point):
    """Ray casting algorithm. Returns True if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class user_app_callback_class(app_callback_class):
    def __init__(self, zone_norm=None, dwell_threshold=10, filter_labels=None):
        super().__init__()
        self.zone_norm = zone_norm  # list of (x,y) normalized tuples or None
        self.dwell_threshold = dwell_threshold  # seconds before loitering alert
        self.filter_labels = set(filter_labels) if filter_labels else None
        # Dwell tracking: {track_id: first_seen_timestamp}
        self.dwell_tracker = {}
        # Track IDs that have already triggered a loitering alert (print once)
        self.dwell_alerted = set()

    def cleanup_stale_ids(self, active_ids):
        """Remove track IDs no longer detected to keep memory bounded."""
        stale = set(self.dwell_tracker.keys()) - active_ids
        for tid in stale:
            del self.dwell_tracker[tid]
            self.dwell_alerted.discard(tid)


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def _print_red(msg: str):
    # ANSI 24-bit red
    print(f"\x1b[38;2;255;0;0m{msg}\x1b[0m")


def _print_green(msg: str):
    # ANSI 24-bit green
    print(f"\x1b[38;2;0;255;0m{msg}\x1b[0m")


def _print_orange(msg: str):
    # ANSI 24-bit orange
    print(f"\x1b[38;2;255;165;0m{msg}\x1b[0m")


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
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0
    now = time.time()

    # Convert zone to pixel coordinates once per frame
    zone_px = None
    if user_data.zone_norm and width is not None and height is not None:
        zone_px = _zone_points_px(user_data.zone_norm, width, height)

    active_ids = set()

    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()

        # Skip detections not in the filter list
        if user_data.filter_labels and label not in user_data.filter_labels:
            continue

        # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        active_ids.add(track_id)

        xyxy = _get_detection_bbox_xyxy(detection, width, height)

        # Zone logic
        in_zone = False
        is_loitering = False
        if zone_px is not None and xyxy is not None:
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            in_zone = _point_in_polygon(zone_px, (cx, cy))

            # Dwell tracking
            if in_zone:
                if track_id not in user_data.dwell_tracker:
                    user_data.dwell_tracker[track_id] = now
                dwell_time = now - user_data.dwell_tracker[track_id]
                if dwell_time >= user_data.dwell_threshold:
                    is_loitering = True
                    if track_id not in user_data.dwell_alerted:
                        user_data.dwell_alerted.add(track_id)
                        _print_orange(
                            f"LOITERING | ID={track_id} | Label={label} | "
                            f"Dwell={dwell_time:.1f}s"
                        )
            else:
                # Left the zone — reset dwell tracking
                user_data.dwell_tracker.pop(track_id, None)
                user_data.dwell_alerted.discard(track_id)

        # Determine color and print
        if is_loitering:
            color = (255, 165, 0)  # orange
        elif in_zone:
            color = (255, 0, 0)  # red
            _print_red(
                f"IN ZONE | ID={track_id} | Label={label} | Conf={confidence:.2f}"
            )
        else:
            color = (0, 255, 0)  # green
            _print_green(
                f"OK | ID={track_id} | Label={label} | Conf={confidence:.2f}"
            )

        # Draw bbox (frame is RGB here)
        if frame is not None and xyxy is not None:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label + track ID above the box
            safe_label = label if label is not None else ""
            label_text = f"{safe_label} ID:{track_id}".strip()
            if is_loitering:
                dwell_time = now - user_data.dwell_tracker.get(track_id, now)
                label_text += f" {dwell_time:.0f}s"
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

        string_to_print += (
            f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
        )
        detection_count += 1

    # Cleanup stale track IDs
    user_data.cleanup_stale_ids(active_ids)

    if user_data.use_frame and frame is not None:
        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw zone polygon
        if zone_px is not None:
            pts = np.array(zone_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return


def main():
    hailo_logger.info("Starting Detection App.")

    parser = get_pipeline_parser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file for detection settings (zone, dwell_threshold, filter_labels). CLI args override config values.",
    )
    parser.add_argument(
        "--zone",
        type=_parse_zone_norm,
        default=None,
        help="Polygon zone as normalized coords: x1,y1,x2,y2,x3,y3,... (0..1, min 3 vertices)",
    )
    parser.add_argument(
        "--dwell-threshold",
        type=float,
        default=None,
        help="Seconds before an in-zone object is flagged as loitering (default: 10)",
    )
    parser.add_argument(
        "--filter-labels",
        type=str,
        nargs="+",
        default=None,
        help="Only show detections with these labels (e.g. --filter-labels person car)",
    )

    # Load YAML config as defaults (CLI args override)
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        hailo_logger.info("Loading detection config from %s", pre_args.config)
        cfg = _load_detection_config(pre_args.config)
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    # Apply default for dwell_threshold after merge
    dwell_threshold = args.dwell_threshold if args.dwell_threshold is not None else 10

    user_data = user_app_callback_class(
        zone_norm=args.zone,
        dwell_threshold=dwell_threshold,
        filter_labels=args.filter_labels,
    )
    app = GStreamerDetectionApp(app_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()

# Running examples:
# Using YAML config file:
#   python3 detection.py --config detection_config_example.yaml --show-frame --input /dev/video8
# Config file with CLI override:
#   python3 detection.py --config detection_config_example.yaml --dwell-threshold 3 --input /dev/video8
# Pure CLI (no config file):
#   python3 detection.py --show-frame --zone 0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9 --dwell-threshold 5 --input /dev/video8
