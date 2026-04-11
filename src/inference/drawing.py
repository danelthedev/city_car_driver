import cv2
import numpy as np

from .detection import _normalize_name, resolve_class_name, scale_bbox_to_frame

# Per-model colour schemes so signs and traffic lights are visually distinct.
_SIGN_BOX_COLOR = (0, 255, 0)
_SIGN_TEXT_COLOR = (0, 255, 0)

_TL_CLASS_COLORS = {
    "red":    (0,   0,   255),
    "yellow": (0,   200, 255),
    "green":  (0,   255, 0),
    "off":    (160, 160, 160),
}
_TL_DEFAULT_COLOR = (255, 255, 255)

_LANE_CLASS_COLORS = {
    "area_drivable":     (70,  180, 70),
    "area_alternative":  (80,  120, 200),
    "lane_crosswalk":    (255, 255, 255),
    "lane_road_curb":    (80,  80,  200),
    "lane_single_white": (245, 245, 245),
    "lane_double_white": (220, 220, 220),
    "lane_single_yellow":(0,   220, 255),
    "lane_double_yellow":(0,   180, 255),
}
_LANE_DEFAULT_COLOR = (180, 180, 180)


def _tl_color(class_name: str):
    return _TL_CLASS_COLORS.get(class_name.lower(), _TL_DEFAULT_COLOR)


def _lane_color(class_name: str):
    return _LANE_CLASS_COLORS.get(_normalize_name(class_name), _LANE_DEFAULT_COLOR)


def _draw_detection(frame, x1, y1, x2, y2, label, box_color, text_color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(
        frame, label, (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2,
    )


def _draw_legend(frame, has_tl_model: bool, has_lane_model: bool):
    items = [("Signs", _SIGN_BOX_COLOR)]
    if has_tl_model:
        items += [
            ("TL: red",    _TL_CLASS_COLORS["red"]),
            ("TL: yellow", _TL_CLASS_COLORS["yellow"]),
            ("TL: green",  _TL_CLASS_COLORS["green"]),
            ("TL: off",    _TL_CLASS_COLORS["off"]),
        ]
    if has_lane_model:
        items += [
            ("Lane: drivable",  _LANE_CLASS_COLORS["area_drivable"]),
            ("Lane: crosswalk", _LANE_CLASS_COLORS["lane_crosswalk"]),
            ("Lane: curb",      _LANE_CLASS_COLORS["lane_road_curb"]),
        ]
    h, w = frame.shape[:2]
    x_start = w - 160
    y_start = 20
    for i, (text, color) in enumerate(items):
        y = y_start + i * 22
        cv2.rectangle(frame, (x_start, y - 12), (x_start + 16, y + 2), color, -1)
        cv2.putText(frame, text, (x_start + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ---------------------------------------------------------------------------
# Lane drawing
# ---------------------------------------------------------------------------

def draw_lane_instances(frame: np.ndarray, lane_instances, alpha: float = 0.35):
    if not lane_instances:
        return frame

    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.0:
        return frame

    overlay = frame.copy()
    for inst in lane_instances:
        mask = inst.get("mask")
        if mask is None:
            continue
        color = _lane_color(inst.get("class_name", ""))
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            continue
        overlay[mask_bool] = color

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    return frame


def build_lane_draw_cache(lane_instances, frame_shape):
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0 or not lane_instances:
        return None

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    color_layer = np.zeros((h, w, 3), dtype=np.uint8)

    for inst in lane_instances:
        mask = inst.get("mask")
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):
            mask_u8 = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask

        mask_bin = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
        if not np.any(mask_bin):
            continue

        color = _lane_color(inst.get("class_name", ""))
        combined_mask = cv2.bitwise_or(combined_mask, mask_bin)
        color_layer[mask_bin > 0] = color

    if not np.any(combined_mask):
        return None

    return {"mask": combined_mask, "color": color_layer}


def draw_lane_from_cache(frame: np.ndarray, lane_cache, alpha: float = 0.35):
    if lane_cache is None:
        return frame

    alpha = max(0.0, min(1.0, float(alpha)))
    mask = lane_cache.get("mask")
    color_layer = lane_cache.get("color")

    if alpha > 0.0 and mask is not None and color_layer is not None and np.any(mask):
        blended = cv2.addWeighted(frame, 1.0 - alpha, color_layer, alpha, 0.0)
        cv2.copyTo(blended, mask, frame)

    return frame


# ---------------------------------------------------------------------------
# Detection overlay caches
# ---------------------------------------------------------------------------

def build_sign_overlay_cache(display_shape, source_shape, detections, detector):
    if not detections:
        return None

    disp_h, disp_w = display_shape[:2]
    src_h, src_w = source_shape[:2]
    if disp_h <= 0 or disp_w <= 0 or src_h <= 0 or src_w <= 0:
        return None

    overlay = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    for (x1, y1, x2, y2, conf, cls) in detections:
        tx1, ty1, tx2, ty2 = scale_bbox_to_frame(x1, y1, x2, y2, src_w, src_h, disp_w, disp_h)
        class_name = resolve_class_name(detector, cls)
        _draw_detection(overlay, tx1, ty1, tx2, ty2, f"{class_name}: {conf:.2f}", _SIGN_BOX_COLOR, _SIGN_TEXT_COLOR)

    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        return None
    return {"overlay": overlay, "mask": mask}


def build_tl_overlay_cache(display_shape, source_shape, detections, tl_detector):
    if not detections:
        return None

    disp_h, disp_w = display_shape[:2]
    src_h, src_w = source_shape[:2]
    if disp_h <= 0 or disp_w <= 0 or src_h <= 0 or src_w <= 0:
        return None

    overlay = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    for (x1, y1, x2, y2, conf, cls) in detections:
        tx1, ty1, tx2, ty2 = scale_bbox_to_frame(x1, y1, x2, y2, src_w, src_h, disp_w, disp_h)
        class_name = resolve_class_name(tl_detector, cls)
        color = _tl_color(class_name)
        _draw_detection(overlay, tx1, ty1, tx2, ty2, f"TL-{class_name}: {conf:.2f}", color, color)

    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        return None
    return {"overlay": overlay, "mask": mask}


def apply_overlay_cache(frame: np.ndarray, overlay_cache):
    if overlay_cache is None:
        return frame

    mask = overlay_cache.get("mask")
    overlay = overlay_cache.get("overlay")
    if mask is None or overlay is None:
        return frame

    cv2.copyTo(overlay, mask, frame)
    return frame
