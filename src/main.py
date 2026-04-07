import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import sys
import threading
import time

import cv2
import mss
import numpy as np
from ultralytics import YOLO

from perception.telemetry import SpeedTelemetryReader
from perception.traffic_signs.factory import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Detection App")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test", "inference"],
        help="Mode to run the application in.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolo", "fasterrcnn", "ssd"],
        help="The detection model string identifier.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/traffic_sign_dataset/dataset.yaml",
        help="Path to the dataset config or root folder for train/test modes.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pre-trained model weights for the primary (sign) model.",
    )

    parser.add_argument(
        "--tl-weights",
        type=str,
        default=None,
        help=(
            "Path to a second YOLO model weights file for traffic-light detection. "
            "When provided, both models run in parallel and results are merged on the same display. "
            "Example: --tl-weights models/yolo11s_bstld.pt"
        ),
    )
    parser.add_argument(
        "--lane-weights",
        type=str,
        default=None,
        help=(
            "Path to a YOLO segmentation model weights file for lane/drivable inference. "
            "Example: --lane-weights models/yolo11s_seg_mini.pt"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (cuda or cpu).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image file for inference (instead of screen capture)."
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.20,
        help="Minimum confidence used during inference.",
    )
    parser.add_argument(
        "--tl-conf-threshold",
        type=float,
        default=None,
        help=(
            "Confidence threshold for the traffic-light model specifically. "
            "Defaults to --conf-threshold if not set."
        ),
    )
    parser.add_argument(
        "--lane-conf-threshold",
        type=float,
        default=None,
        help=(
            "Confidence threshold for the lane segmentation model specifically. "
            "Defaults to --conf-threshold if not set."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--inference-scale",
        type=float,
        default=1.0,
        help="Scale applied before inference only (0 < value <= 1).",
    )
    parser.add_argument(
        "--lane-imgsz",
        type=int,
        default=None,
        help="YOLO inference image size for lane segmentation model. Defaults to --imgsz.",
    )
    parser.add_argument(
        "--lane-inference-scale",
        type=float,
        default=None,
        help="Scale applied before lane inference only (0 < value <= 1). Defaults to --inference-scale.",
    )
    parser.add_argument(
        "--lane-draw-contours",
        action="store_true",
        help="Draw lane contour outlines. Disabled by default for better FPS.",
    )
    parser.add_argument(
        "--sync-inference",
        action="store_true",
        help="Run inference synchronously. Default behavior uses async inference for smoother display.",
    )
    parser.set_defaults(lane_priority=True)
    parser.add_argument(
        "--lane-priority",
        dest="lane_priority",
        action="store_true",
        help="Prioritize lane responsiveness by throttling sign/TL async submissions when lane model is enabled.",
    )
    parser.add_argument(
        "--no-lane-priority",
        dest="lane_priority",
        action="store_false",
        help="Disable lane-priority scheduling.",
    )
    parser.add_argument(
        "--sign-submit-interval-ms",
        type=float,
        default=0.0,
        help="Minimum time between sign async submissions in milliseconds (0 = unlimited).",
    )
    parser.add_argument(
        "--tl-submit-interval-ms",
        type=float,
        default=0.0,
        help="Minimum time between traffic-light async submissions in milliseconds (0 = unlimited).",
    )
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=1,
        help="Monitor index for screen capture (mss indexing: 1..N).",
    )
    parser.add_argument(
        "--async-capture",
        action="store_true",
        help="Capture frames in a background thread and consume the latest frame in the render loop.",
    )
    parser.add_argument(
        "--max-draw-detections",
        type=int,
        default=40,
        help="Maximum detections to draw per frame (set <= 0 to draw all).",
    )
    parser.add_argument(
        "--min-confirm-frames",
        type=int,
        default=2,
        help="Minimum consecutive inference updates required before drawing a detection.",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=2,
        help="How many inference updates a confirmed detection can be missing before removal.",
    )
    parser.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.35,
        help="IoU threshold for matching detections across inference updates.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1400,
        help="Inference display window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Inference display window height in pixels.",
    )
    parser.add_argument(
        "--lane-mask-alpha",
        type=float,
        default=0.35,
        help="Transparency for lane segmentation overlays (0.0 to 1.0).",
    )
    parser.add_argument(
        "--intersection-threshold",
        type=float,
        default=0.55,
        help="Threshold used for the intersection-ahead score from lane geometry cues.",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print averaged stage timings every N frames for FPS bottleneck analysis.",
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=120,
        help="How many frames to aggregate before printing profiling statistics.",
    )
    parser.add_argument(
        "--speed-telemetry",
        action="store_true",
        help="Enable speed extraction from process memory via the perception telemetry module.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# Per-model colour schemes so signs and traffic lights are visually distinct.
# Signs: green boxes.  Traffic lights: colour-coded by class name.
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
    "area_drivable": (70, 180, 70),
    "area_alternative": (80, 120, 200),
    "lane_crosswalk": (255, 255, 255),
    "lane_road_curb": (80, 80, 200),
    "lane_single_white": (245, 245, 245),
    "lane_double_white": (220, 220, 220),
    "lane_single_yellow": (0, 220, 255),
    "lane_double_yellow": (0, 180, 255),
}
_LANE_DEFAULT_COLOR = (180, 180, 180)


def _tl_color(class_name: str):
    return _TL_CLASS_COLORS.get(class_name.lower(), _TL_DEFAULT_COLOR)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _lane_color(class_name: str):
    return _LANE_CLASS_COLORS.get(_normalize_name(class_name), _LANE_DEFAULT_COLOR)


def _draw_detection(frame, x1, y1, x2, y2, label, box_color, text_color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(
        frame, label, (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2,
    )


def _draw_legend(frame, has_tl_model: bool, has_lane_model: bool):
    """Draw a small legend in the top-right corner."""
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
            ("Lane: drivable", _LANE_CLASS_COLORS["area_drivable"]),
            ("Lane: crosswalk", _LANE_CLASS_COLORS["lane_crosswalk"]),
            ("Lane: curb", _LANE_CLASS_COLORS["lane_road_curb"]),
        ]
    h, w = frame.shape[:2]
    x_start = w - 160
    y_start = 20
    for i, (text, color) in enumerate(items):
        y = y_start + i * 22
        cv2.rectangle(frame, (x_start, y - 12), (x_start + 16, y + 2), color, -1)
        cv2.putText(frame, text, (x_start + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def run_training(args, detector):
    print(f"Starting training for {args.model} on {args.data} for {args.epochs} epochs.")
    results = detector.train(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)

    if args.model == "yolo" and hasattr(detector.model.trainer, "best"):
        best_weights_path = detector.model.trainer.best
        if best_weights_path and os.path.exists(best_weights_path):
            os.makedirs("models", exist_ok=True)
            base_model_name = os.path.splitext(os.path.basename(args.weights or "yolov8n"))[0]
            target_path = os.path.join("models", f"{base_model_name}_best.pt")
            shutil.copy2(best_weights_path, target_path)
            print(f"\n[INFO] Best {base_model_name} model copied to: {target_path}")
        else:
            print("\n[WARNING] Could not locate best.pt to copy to models directory.")


def run_evaluation(args, detector):
    if not args.weights:
        print("Warning: No weights provided for evaluation.")
    print(f"Starting evaluation for {args.model} on {args.data}.")
    metrics = detector.evaluate(data_path=args.data)
    print("Metrics:", metrics)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def resolve_class_name(detector, class_id: int) -> str:
    model = getattr(detector, "model", None)
    names = getattr(model, "names", None) if model is not None else None

    if isinstance(names, dict):
        if class_id in names:
            return str(names[class_id])
        str_id = str(class_id)
        if str_id in names:
            return str(names[str_id])
    elif isinstance(names, (list, tuple)):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    return f"class-{class_id}"


def resize_to_fit(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    image_h, image_w = image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return image

    scale = min(float(target_width) / float(image_w), float(target_height) / float(image_h))
    scale = max(scale, 0.01)
    resized_w = max(1, int(round(image_w * scale)))
    resized_h = max(1, int(round(image_h * scale)))

    if resized_w == image_w and resized_h == image_h:
        return image
    return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)


def scale_detections_to_original(detections, scale: float, frame_width: int, frame_height: int):
    if not detections or scale == 1.0:
        return detections

    inv_scale = 1.0 / scale
    max_x = max(frame_width - 1, 0)
    max_y = max(frame_height - 1, 0)
    scaled = []

    for (x1, y1, x2, y2, conf, cls) in detections:
        sx1 = max(0, min(max_x, int(round(x1 * inv_scale))))
        sy1 = max(0, min(max_y, int(round(y1 * inv_scale))))
        sx2 = max(0, min(max_x, int(round(x2 * inv_scale))))
        sy2 = max(0, min(max_y, int(round(y2 * inv_scale))))

        if sx2 <= sx1 or sy2 <= sy1:
            continue
        scaled.append((sx1, sy1, sx2, sy2, conf, cls))

    return scaled


def scale_bbox_to_frame(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
):
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return x1, y1, x2, y2

    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)

    tx1 = int(round(x1 * sx))
    ty1 = int(round(y1 * sy))
    tx2 = int(round(x2 * sx))
    ty2 = int(round(y2 * sy))

    tx1 = max(0, min(dst_width - 1, tx1))
    ty1 = max(0, min(dst_height - 1, ty1))
    tx2 = max(0, min(dst_width - 1, tx2))
    ty2 = max(0, min(dst_height - 1, ty2))

    if tx2 <= tx1:
        tx2 = min(dst_width - 1, tx1 + 1)
    if ty2 <= ty1:
        ty2 = min(dst_height - 1, ty1 + 1)

    return tx1, ty1, tx2, ty2


def run_detection_once(detector, frame: np.ndarray, confidence_threshold: float, image_size: int, inference_scale: float):
    start_time = time.perf_counter()
    frame_h, frame_w = frame.shape[:2]

    scale = float(inference_scale)
    if scale <= 0.0:
        scale = 1.0

    inference_frame = frame
    if scale != 1.0:
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        inference_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

    detections = detector.predict(
        inference_frame,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
    )
    detections = scale_detections_to_original(detections, scale, frame_w, frame_h)
    duration = max(time.perf_counter() - start_time, 1e-6)
    return detections, duration


def run_lane_segmentation_once(lane_model, frame: np.ndarray, confidence_threshold: float, image_size: int, inference_scale: float):
    start_time = time.perf_counter()
    frame_h, frame_w = frame.shape[:2]

    scale = float(inference_scale)
    if scale <= 0.0:
        scale = 1.0

    inference_frame = frame
    if scale != 1.0:
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        inference_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

    results = lane_model(
        inference_frame,
        conf=confidence_threshold,
        imgsz=max(320, int(image_size)),
        verbose=False,
    )
    result = results[0]

    lane_instances = []
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)

    if masks is not None and boxes is not None and len(masks.data) > 0 and len(boxes) > 0:
        mask_data = masks.data.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(mask_data),), dtype=np.float32)
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(mask_data),), dtype=np.float32)

        for idx in range(min(len(mask_data), len(confs), len(classes))):
            raw_mask = (mask_data[idx] > 0.5).astype(np.uint8)
            if raw_mask.sum() <= 0:
                continue

            if raw_mask.shape[1] != frame_w or raw_mask.shape[0] != frame_h:
                raw_mask = cv2.resize(raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

            class_id = int(classes[idx])
            class_name = str(result.names.get(class_id, f"class-{class_id}")) if isinstance(result.names, dict) else str(result.names[class_id]) if isinstance(result.names, (list, tuple)) and 0 <= class_id < len(result.names) else f"class-{class_id}"

            lane_instances.append(
                {
                    "mask": raw_mask,
                    "class_id": class_id,
                    "class_name": class_name,
                    "conf": float(confs[idx]),
                }
            )

    duration = max(time.perf_counter() - start_time, 1e-6)
    return lane_instances, duration


def draw_lane_instances(frame: np.ndarray, lane_instances, alpha: float = 0.35, draw_contours: bool = False):
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

        if draw_contours:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    return frame


def build_lane_draw_cache(lane_instances, frame_shape, draw_contours: bool = False):
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0 or not lane_instances:
        return None

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    color_layer = np.zeros((h, w, 3), dtype=np.uint8)
    contour_items = []

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

        if draw_contours:
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour_items.append((contours, color))

    if not np.any(combined_mask):
        return None

    return {
        "mask": combined_mask,
        "color": color_layer,
        "contours": contour_items,
    }


def draw_lane_from_cache(frame: np.ndarray, lane_cache, alpha: float = 0.35, draw_contours: bool = False):
    if lane_cache is None:
        return frame

    alpha = max(0.0, min(1.0, float(alpha)))
    mask = lane_cache.get("mask")
    color_layer = lane_cache.get("color")

    if alpha > 0.0 and mask is not None and color_layer is not None and np.any(mask):
        blended = cv2.addWeighted(frame, 1.0 - alpha, color_layer, alpha, 0.0)
        cv2.copyTo(blended, mask, frame)

    if draw_contours:
        for contours, color in lane_cache.get("contours", []):
            cv2.drawContours(frame, contours, -1, color, 2)

    return frame


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
        _draw_detection(
            overlay,
            tx1,
            ty1,
            tx2,
            ty2,
            f"{class_name}: {conf:.2f}",
            _SIGN_BOX_COLOR,
            _SIGN_TEXT_COLOR,
        )

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
        _draw_detection(
            overlay,
            tx1,
            ty1,
            tx2,
            ty2,
            f"TL-{class_name}: {conf:.2f}",
            color,
            color,
        )

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


def compute_intersection_status(lane_instances, frame_shape, threshold: float = 0.55):
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, False, []

    x1 = int(0.20 * w)
    x2 = int(0.80 * w)
    y1 = int(0.35 * h)
    y2 = int(0.90 * h)
    roi_w = max(x2 - x1, 1)
    roi_h = max(y2 - y1, 1)
    roi_area = float(roi_w * roi_h)

    drivable_mask = np.zeros((h, w), dtype=np.uint8)
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    crosswalk_mask = np.zeros((h, w), dtype=np.uint8)

    for inst in lane_instances:
        mask = inst.get("mask")
        if mask is None:
            continue

        cls = _normalize_name(inst.get("class_name", ""))
        if cls in {"area_drivable", "area_alternative"}:
            drivable_mask = np.maximum(drivable_mask, mask)
        elif cls == "lane_crosswalk":
            crosswalk_mask = np.maximum(crosswalk_mask, mask)
        elif cls.startswith("lane_"):
            lane_mask = np.maximum(lane_mask, mask)

    drivable_roi = drivable_mask[y1:y2, x1:x2]
    lane_roi = lane_mask[y1:y2, x1:x2]
    crosswalk_roi = crosswalk_mask[y1:y2, x1:x2]

    drivable_ratio = float(np.count_nonzero(drivable_roi)) / roi_area
    lane_ratio = float(np.count_nonzero(lane_roi)) / roi_area
    crosswalk_ratio = float(np.count_nonzero(crosswalk_roi)) / roi_area

    upper = lane_mask[int(0.35 * h):int(0.60 * h), x1:x2]
    num_components, _ = cv2.connectedComponents((upper > 0).astype(np.uint8))
    branching_components = max(0, int(num_components) - 1)

    score = 0.0
    reasons = []

    if crosswalk_ratio > 0.012:
        score += 0.45
        reasons.append("crosswalk")
    if drivable_ratio > 0.20:
        score += 0.25
        reasons.append("wide-drivable")
    if lane_ratio > 0.02 and branching_components >= 3:
        score += 0.30
        reasons.append("lane-branching")

    score = max(0.0, min(1.0, score))
    is_intersection = score >= float(threshold)
    return score, is_intersection, reasons


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area) / float(union)


def apply_temporal_consensus(
    detections,
    tracks,
    next_track_id: int,
    min_confirm_frames: int,
    max_missing_frames: int,
    match_iou_threshold: float,
):
    sorted_detections = sorted(detections, key=lambda item: item[4], reverse=True)
    matched_track_ids = set()

    for (x1, y1, x2, y2, conf, cls) in sorted_detections:
        best_track_id = None
        best_iou = match_iou_threshold

        for track_id, track in tracks.items():
            if track_id in matched_track_ids:
                continue
            if int(track["cls"]) != int(cls):
                continue

            iou = compute_iou((x1, y1, x2, y2), track["bbox"])
            if iou >= best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            tracks[next_track_id] = {
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(conf),
                "cls": int(cls),
                "hits": 1,
                "missed": 0,
            }
            matched_track_ids.add(next_track_id)
            next_track_id += 1
        else:
            track = tracks[best_track_id]
            track["bbox"] = (int(x1), int(y1), int(x2), int(y2))
            track["conf"] = float(conf)
            track["hits"] = int(track["hits"]) + 1
            track["missed"] = 0
            matched_track_ids.add(best_track_id)

    stale_track_ids = []
    stable_detections = []

    for track_id, track in tracks.items():
        if track_id not in matched_track_ids:
            track["missed"] = int(track["missed"]) + 1

        if int(track["missed"]) > max_missing_frames:
            stale_track_ids.append(track_id)
            continue

        if int(track["hits"]) >= min_confirm_frames:
            tx1, ty1, tx2, ty2 = track["bbox"]
            stable_detections.append((tx1, ty1, tx2, ty2, float(track["conf"]), int(track["cls"])))

    for track_id in stale_track_ids:
        del tracks[track_id]

    stable_detections.sort(key=lambda item: item[4], reverse=True)
    return stable_detections, next_track_id


# ---------------------------------------------------------------------------
# Inference — screen capture (single or dual model)
# ---------------------------------------------------------------------------

def run_inference_screen_capture(args, detector, tl_detector=None, lane_model=None):
    print("Starting screen capture inference...")
    print("Press 'q' in the window to stop.")

    has_tl = tl_detector is not None
    has_lane = lane_model is not None
    lane_priority_enabled = bool(args.lane_priority and has_lane)
    tl_conf = args.tl_conf_threshold if args.tl_conf_threshold is not None else args.conf_threshold
    lane_conf = args.lane_conf_threshold if args.lane_conf_threshold is not None else args.conf_threshold
    lane_imgsz = args.lane_imgsz if args.lane_imgsz is not None else args.imgsz
    lane_scale = args.lane_inference_scale if args.lane_inference_scale is not None else args.inference_scale
    sign_submit_interval_ms = max(0.0, float(args.sign_submit_interval_ms))
    tl_submit_interval_ms = max(0.0, float(args.tl_submit_interval_ms))

    if lane_priority_enabled:
        # If user did not force explicit intervals, apply lane-first defaults.
        if sign_submit_interval_ms <= 0.0:
            sign_submit_interval_ms = 80.0
        if has_tl and tl_submit_interval_ms <= 0.0:
            tl_submit_interval_ms = 80.0

    inference_mode = "sync" if args.sync_inference else "async"
    capture_mode = "async" if args.async_capture else "sync"
    print(
        f"Inference settings: conf={args.conf_threshold:.2f}, "
        + (f"tl_conf={tl_conf:.2f}, " if has_tl else "")
        + (f"lane_conf={lane_conf:.2f}, " if has_lane else "")
        + f"imgsz={args.imgsz}, infer_scale={args.inference_scale:.2f}, "
        + (f"lane_imgsz={lane_imgsz}, lane_scale={lane_scale:.2f}, " if has_lane else "")
        + f"mode={inference_mode}, capture={capture_mode}, lane_priority={lane_priority_enabled}, confirm={args.min_confirm_frames}, "
        f"max_miss={args.max_missing_frames}, track_iou={args.track_iou_threshold:.2f}, "
        f"window={args.window_width}x{args.window_height}"
    )
    if not args.sync_inference and (sign_submit_interval_ms > 0.0 or (has_tl and tl_submit_interval_ms > 0.0)):
        print(
            f"[INFO] Async submit intervals: sign={sign_submit_interval_ms:.1f}ms"
            + (f", tl={tl_submit_interval_ms:.1f}ms" if has_tl else "")
        )
    if args.profile_stages:
        print(f"[INFO] Stage profiling enabled. Printing every {max(1, args.profile_every)} frames.")
    if has_tl:
        print("[INFO] Dual-model mode: sign model + traffic-light model running in parallel.")
    if has_lane:
        print("[INFO] Lane segmentation enabled.")

    sct = mss.mss()
    monitor_index = args.monitor_index
    if monitor_index < 1 or monitor_index >= len(sct.monitors):
        fallback_index = 1 if len(sct.monitors) > 1 else 0
        print(f"[WARNING] Invalid monitor index {monitor_index}. Falling back to {fallback_index}.")
        monitor_index = fallback_index

    monitor = sct.monitors[monitor_index]
    print(
        f"Capture monitor {monitor_index}: "
        f"{monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})"
    )

    window_name = f"{args.model} - Real-time Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    # Per-model state
    sign_detections_raw = []
    sign_stable = []
    sign_to_draw = []
    sign_overlay_cache = None
    sign_overlay_shape = None
    sign_tracks = {}
    sign_next_id = 1
    sign_raw_count = 0
    sign_infer_duration = 0.0

    tl_detections_raw = []
    tl_stable = []
    tl_to_draw = []
    tl_overlay_cache = None
    tl_overlay_shape = None
    tl_tracks = {}
    tl_next_id = 1
    tl_raw_count = 0
    tl_infer_duration = 0.0

    lane_instances = []
    lane_draw_cache = None
    lane_draw_cache_shape = None
    lane_infer_duration = 0.0
    lane_intersection_score = 0.0
    lane_intersection = False
    lane_intersection_reasons = []

    speed_telemetry_reader = None
    speed_kmh = None
    speed_last_update_ts = 0.0
    speed_process_name = "starter.exe"
    speed_module_name = "pdd.dll"
    speed_offset = 0xE322B0
    speed_poll_interval_ms = 50.0
    speed_stale_timeout_s = 0.50
    if args.speed_telemetry:
        try:
            speed_telemetry_reader = SpeedTelemetryReader(
                process_name=speed_process_name,
                module_name=speed_module_name,
                speed_offset=speed_offset,
                poll_interval_ms=speed_poll_interval_ms,
            )
            print(
                "[INFO] Speed telemetry enabled: "
                + f"process={speed_process_name}, module={speed_module_name}, "
                + f"offset=0x{int(speed_offset):X}, poll={float(speed_poll_interval_ms):.1f}ms"
            )
        except Exception as exc:
            print(f"[WARNING] Failed to initialize speed telemetry: {exc}")
            speed_telemetry_reader = None

    display_fps_ema = 0.0
    sign_fps_ema = 0.0
    tl_fps_ema = 0.0
    lane_fps_ema = 0.0

    profile_totals = {
        "capture": 0.0,
        "sign_infer": 0.0,
        "tl_infer": 0.0,
        "lane_infer": 0.0,
        "sign_consensus": 0.0,
        "tl_consensus": 0.0,
        "intersection": 0.0,
        "draw": 0.0,
        "display": 0.0,
        "loop": 0.0,
    }
    frame_count = 0
    sign_stale_frames = 0
    tl_stale_frames = 0
    lane_stale_frames = 0
    capture_reuse_frames = 0
    sign_update_count = 0
    tl_update_count = 0
    lane_update_count = 0

    # Each model gets its own single-thread executor for async mode
    sign_worker = None
    tl_worker = None
    lane_worker = None
    sign_pending = None
    tl_pending = None
    lane_pending = None
    sign_last_submit_time = 0.0
    tl_last_submit_time = 0.0

    if not args.sync_inference:
        sign_worker = ThreadPoolExecutor(max_workers=1)
        if has_tl:
            tl_worker = ThreadPoolExecutor(max_workers=1)
        if has_lane:
            lane_worker = ThreadPoolExecutor(max_workers=1)

    capture_lock = None
    latest_frame = None
    latest_display_frame = None
    last_async_frame = None
    last_async_display_frame = None
    capture_stop_event = None
    capture_thread = None
    capture_error = None

    if args.async_capture:
        capture_lock = threading.Lock()
        capture_stop_event = threading.Event()

        def capture_loop():
            nonlocal latest_frame, latest_display_frame, capture_error
            local_capture_lock = capture_lock
            assert local_capture_lock is not None
            try:
                with mss.mss() as capture_sct:
                    while not capture_stop_event.is_set():
                        grabbed = np.array(capture_sct.grab(monitor))
                        bgr_frame = cv2.cvtColor(grabbed, cv2.COLOR_BGRA2BGR)
                        disp_base = resize_to_fit(bgr_frame, args.window_width, args.window_height)
                        with local_capture_lock:
                            latest_frame = bgr_frame
                            latest_display_frame = disp_base
            except Exception as exc:
                capture_error = exc
                capture_stop_event.set()

        capture_thread = threading.Thread(target=capture_loop, name="screen-capture-thread", daemon=True)
        capture_thread.start()
        print("[INFO] Async capture enabled (latest-frame mode).")

    try:
        while True:
            loop_start = time.perf_counter()

            capture_start = time.perf_counter()
            stop_requested = False

            if not args.async_capture:
                screen_img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
                disp_base = resize_to_fit(frame, args.window_width, args.window_height)
            else:
                assert capture_lock is not None
                frame = None
                disp_base = None

                if capture_error is not None:
                    raise RuntimeError(f"Background capture failed: {capture_error}")

                with capture_lock:
                    if latest_frame is not None:
                        frame = latest_frame
                        disp_base = latest_display_frame
                        latest_frame = None
                        latest_display_frame = None

                if frame is None and last_async_frame is not None:
                    frame = last_async_frame
                    disp_base = last_async_display_frame
                    capture_reuse_frames += 1

                while frame is None:
                    if capture_error is not None:
                        raise RuntimeError(f"Background capture failed: {capture_error}")

                    with capture_lock:
                        if latest_frame is not None:
                            frame = latest_frame
                            disp_base = latest_display_frame
                            latest_frame = None
                            latest_display_frame = None

                    if frame is None:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_requested = True
                            break
                        time.sleep(0.001)

                if stop_requested:
                    break

                last_async_frame = frame
                last_async_display_frame = disp_base

            assert frame is not None
            if disp_base is None:
                disp_base = resize_to_fit(frame, args.window_width, args.window_height)
            capture_duration = max(time.perf_counter() - capture_start, 0.0)

            # Always draw on a fresh frame copy to avoid cumulative alpha blending
            # when async capture reuses the same base frame across display ticks.
            disp_frame = disp_base.copy()

            # In async mode, all workers consume the same immutable frame copy.
            # This avoids one full image copy per model per frame.
            async_input_frame = frame.copy() if sign_worker is not None else frame
            sign_consensus_duration = 0.0
            tl_consensus_duration = 0.0
            intersection_duration = 0.0
            lane_updated = False
            sign_updated = False
            tl_updated = False

            # ----------------------------------------------------------
            # Sign model
            # ----------------------------------------------------------
            if sign_worker is None:
                sign_detections_raw, sign_infer_duration = run_detection_once(
                    detector, frame, args.conf_threshold, args.imgsz, args.inference_scale
                )
                sign_update_count += 1
                sign_raw_count = len(sign_detections_raw)
                sign_consensus_start = time.perf_counter()
                sign_stable, sign_next_id = apply_temporal_consensus(
                    sign_detections_raw, sign_tracks, sign_next_id,
                    args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                )
                sign_consensus_duration = max(time.perf_counter() - sign_consensus_start, 0.0)
                sign_updated = True
            else:
                if sign_pending is not None and not sign_pending.done():
                    sign_stale_frames += 1

                if sign_pending is not None and sign_pending.done():
                    try:
                        sign_detections_raw, sign_infer_duration = sign_pending.result()
                        sign_update_count += 1
                        sign_raw_count = len(sign_detections_raw)
                        sign_consensus_start = time.perf_counter()
                        sign_stable, sign_next_id = apply_temporal_consensus(
                            sign_detections_raw, sign_tracks, sign_next_id,
                            args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                        )
                        sign_consensus_duration = max(time.perf_counter() - sign_consensus_start, 0.0)
                        sign_updated = True
                    except Exception as exc:
                        print(f"[ERROR] Sign model inference failed: {exc}")
                        sign_detections_raw = []
                        sign_stable = []
                        sign_tracks.clear()
                        sign_updated = True
                    sign_pending = None

                if sign_pending is None:
                    now_ts = time.perf_counter()
                    if ((now_ts - sign_last_submit_time) * 1000.0) >= sign_submit_interval_ms:
                        sign_pending = sign_worker.submit(
                            run_detection_once,
                            detector,
                            async_input_frame,
                            args.conf_threshold,
                            args.imgsz,
                            args.inference_scale,
                        )
                        sign_last_submit_time = now_ts

            # ----------------------------------------------------------
            # Traffic light model (if provided)
            # ----------------------------------------------------------
            if has_tl:
                if tl_worker is None:
                    tl_detections_raw, tl_infer_duration = run_detection_once(
                        tl_detector, frame, tl_conf, args.imgsz, args.inference_scale
                    )
                    tl_update_count += 1
                    tl_raw_count = len(tl_detections_raw)
                    tl_consensus_start = time.perf_counter()
                    tl_stable, tl_next_id = apply_temporal_consensus(
                        tl_detections_raw, tl_tracks, tl_next_id,
                        args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                    )
                    tl_consensus_duration = max(time.perf_counter() - tl_consensus_start, 0.0)
                    tl_updated = True
                else:
                    if tl_pending is not None and not tl_pending.done():
                        tl_stale_frames += 1

                    if tl_pending is not None and tl_pending.done():
                        try:
                            tl_detections_raw, tl_infer_duration = tl_pending.result()
                            tl_update_count += 1
                            tl_raw_count = len(tl_detections_raw)
                            tl_consensus_start = time.perf_counter()
                            tl_stable, tl_next_id = apply_temporal_consensus(
                                tl_detections_raw, tl_tracks, tl_next_id,
                                args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                            )
                            tl_consensus_duration = max(time.perf_counter() - tl_consensus_start, 0.0)
                            tl_updated = True
                        except Exception as exc:
                            print(f"[ERROR] Traffic light model inference failed: {exc}")
                            tl_detections_raw = []
                            tl_stable = []
                            tl_tracks.clear()
                            tl_updated = True
                        tl_pending = None

                    if tl_pending is None:
                        now_ts = time.perf_counter()
                        if ((now_ts - tl_last_submit_time) * 1000.0) >= tl_submit_interval_ms:
                            tl_pending = tl_worker.submit(
                                run_detection_once,
                                tl_detector,
                                async_input_frame,
                                tl_conf,
                                args.imgsz,
                                args.inference_scale,
                            )
                            tl_last_submit_time = now_ts

            # ----------------------------------------------------------
            # Lane segmentation model (if provided)
            # ----------------------------------------------------------
            if has_lane:
                if lane_worker is None:
                    lane_instances, lane_infer_duration = run_lane_segmentation_once(
                        lane_model, frame, lane_conf, lane_imgsz, lane_scale
                    )
                    lane_update_count += 1
                    lane_updated = True
                else:
                    if lane_pending is not None and not lane_pending.done():
                        lane_stale_frames += 1

                    if lane_pending is not None and lane_pending.done():
                        try:
                            lane_instances, lane_infer_duration = lane_pending.result()
                            lane_update_count += 1
                            lane_updated = True
                        except Exception as exc:
                            print(f"[ERROR] Lane model inference failed: {exc}")
                            lane_instances = []
                            lane_updated = True
                        lane_pending = None

                    if lane_pending is None:
                        lane_pending = lane_worker.submit(
                            run_lane_segmentation_once,
                            lane_model,
                            async_input_frame,
                            lane_conf,
                            lane_imgsz,
                            lane_scale,
                        )

                cache_needs_refresh = (
                    lane_draw_cache is None
                    or lane_updated
                )
                if cache_needs_refresh:
                    # Cache rebuild is deferred until display size is known for this frame.
                    lane_draw_cache_shape = None

                if lane_updated:
                    intersection_start = time.perf_counter()
                    lane_intersection_score, lane_intersection, lane_intersection_reasons = compute_intersection_status(
                        lane_instances,
                        frame.shape,
                        threshold=args.intersection_threshold,
                    )
                    intersection_duration = max(time.perf_counter() - intersection_start, 0.0)

            draw_start = time.perf_counter()
            disp_h, disp_w = disp_frame.shape[:2]

            if speed_telemetry_reader is not None:
                now_ts = time.perf_counter()
                latest_speed = speed_telemetry_reader.read_speed_if_due(now_ts)
                if latest_speed is not None:
                    speed_kmh = float(latest_speed)
                    speed_last_update_ts = now_ts
                elif speed_kmh is not None and (now_ts - speed_last_update_ts) > speed_stale_timeout_s:
                    speed_kmh = None

            if has_lane:
                cache_needs_refresh = (
                    lane_draw_cache is None
                    or lane_updated
                    or lane_draw_cache_shape != (disp_h, disp_w)
                )
                if cache_needs_refresh:
                    lane_draw_cache = build_lane_draw_cache(
                        lane_instances,
                        disp_frame.shape,
                        draw_contours=args.lane_draw_contours,
                    )
                    lane_draw_cache_shape = (disp_h, disp_w)

                draw_lane_from_cache(
                    disp_frame,
                    lane_draw_cache,
                    alpha=args.lane_mask_alpha,
                    draw_contours=args.lane_draw_contours,
                )

            # ----------------------------------------------------------
            # Draw sign detections (green)
            # ----------------------------------------------------------
            sign_to_draw = sign_stable
            if args.max_draw_detections > 0:
                sign_to_draw = sign_to_draw[: args.max_draw_detections]
            sign_cache_needs_refresh = (
                sign_overlay_cache is None
                or sign_updated
                or sign_overlay_shape != (disp_h, disp_w)
            )
            if sign_cache_needs_refresh:
                sign_overlay_cache = build_sign_overlay_cache(
                    disp_frame.shape,
                    frame.shape,
                    sign_to_draw,
                    detector,
                )
                sign_overlay_shape = (disp_h, disp_w)

            apply_overlay_cache(disp_frame, sign_overlay_cache)

            # ----------------------------------------------------------
            # Draw traffic light detections (colour-coded)
            # ----------------------------------------------------------
            if has_tl:
                tl_to_draw = tl_stable
                if args.max_draw_detections > 0:
                    tl_to_draw = tl_to_draw[: args.max_draw_detections]
                tl_cache_needs_refresh = (
                    tl_overlay_cache is None
                    or tl_updated
                    or tl_overlay_shape != (disp_h, disp_w)
                )
                if tl_cache_needs_refresh:
                    tl_overlay_cache = build_tl_overlay_cache(
                        disp_frame.shape,
                        frame.shape,
                        tl_to_draw,
                        tl_detector,
                    )
                    tl_overlay_shape = (disp_h, disp_w)

                apply_overlay_cache(disp_frame, tl_overlay_cache)

            # ----------------------------------------------------------
            # HUD
            # ----------------------------------------------------------
            loop_duration = max(time.perf_counter() - loop_start, 1e-6)
            display_fps = 1.0 / loop_duration
            sign_fps = 1.0 / sign_infer_duration if sign_infer_duration > 0.0 else 0.0
            tl_fps = 1.0 / tl_infer_duration if tl_infer_duration > 0.0 else 0.0
            lane_fps = 1.0 / lane_infer_duration if lane_infer_duration > 0.0 else 0.0

            display_fps_ema = display_fps if display_fps_ema == 0.0 else 0.9 * display_fps_ema + 0.1 * display_fps
            if sign_fps > 0.0:
                sign_fps_ema = sign_fps if sign_fps_ema == 0.0 else 0.9 * sign_fps_ema + 0.1 * sign_fps
            if tl_fps > 0.0:
                tl_fps_ema = tl_fps if tl_fps_ema == 0.0 else 0.9 * tl_fps_ema + 0.1 * tl_fps
            if lane_fps > 0.0:
                lane_fps_ema = lane_fps if lane_fps_ema == 0.0 else 0.9 * lane_fps_ema + 0.1 * lane_fps

            mode_tag = "SYNC" if args.sync_inference else "ASYNC"
            cv2.putText(disp_frame, f"Display FPS: {display_fps_ema:.1f} | Sign FPS: {sign_fps_ema:.1f}" + (f" | TL FPS: {tl_fps_ema:.1f}" if has_tl else "") + (f" | Lane FPS: {lane_fps_ema:.1f}" if has_lane else ""), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            cv2.putText(disp_frame, f"Mode: {mode_tag} | Scale: {args.inference_scale:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(disp_frame, f"Signs raw/stable: {sign_raw_count}/{len(sign_to_draw)}" + (f"  |  TL raw/stable: {tl_raw_count}/{len(tl_to_draw) if has_tl else 0}" if has_tl else ""), (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            if has_lane:
                status_text = f"Intersection: {'YES' if lane_intersection else 'no'} ({lane_intersection_score:.2f})"
                if lane_intersection_reasons:
                    status_text += " | " + ",".join(lane_intersection_reasons)
                cv2.putText(disp_frame, status_text, (20, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 220) if lane_intersection else (80, 180, 80), 2)
            if speed_telemetry_reader is not None:
                speed_text = f"Speed(mem): {speed_kmh:.1f} km/h" if speed_kmh is not None else "Speed(mem): --"
                cv2.putText(disp_frame, speed_text, (20, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            _draw_legend(disp_frame, has_tl, has_lane)
            draw_duration = max(time.perf_counter() - draw_start, 0.0)

            display_start = time.perf_counter()
            cv2.imshow(window_name, disp_frame)
            key = cv2.waitKey(1)
            display_duration = max(time.perf_counter() - display_start, 0.0)

            loop_duration = max(time.perf_counter() - loop_start, 1e-6)

            if args.profile_stages:
                frame_count += 1
                profile_totals["capture"] += capture_duration
                profile_totals["sign_infer"] += max(sign_infer_duration, 0.0)
                profile_totals["tl_infer"] += max(tl_infer_duration, 0.0)
                profile_totals["lane_infer"] += max(lane_infer_duration, 0.0)
                profile_totals["sign_consensus"] += sign_consensus_duration
                profile_totals["tl_consensus"] += tl_consensus_duration
                profile_totals["intersection"] += intersection_duration
                profile_totals["draw"] += draw_duration
                profile_totals["display"] += display_duration
                profile_totals["loop"] += loop_duration

                if frame_count % max(1, args.profile_every) == 0:
                    divisor = float(frame_count)
                    avg_loop_ms = (profile_totals["loop"] / divisor) * 1000.0
                    avg_fps = (1.0 / (profile_totals["loop"] / divisor)) if profile_totals["loop"] > 0.0 else 0.0
                    elapsed_s = max(profile_totals["loop"], 1e-6)
                    sign_stale_pct = (100.0 * sign_stale_frames / divisor) if sign_worker is not None else 0.0
                    tl_stale_pct = (100.0 * tl_stale_frames / divisor) if tl_worker is not None else 0.0
                    lane_stale_pct = (100.0 * lane_stale_frames / divisor) if lane_worker is not None else 0.0
                    capture_reuse_pct = (100.0 * capture_reuse_frames / divisor) if args.async_capture else 0.0
                    sign_updates_hz = (float(sign_update_count) / elapsed_s) if sign_worker is not None else 0.0
                    tl_updates_hz = (float(tl_update_count) / elapsed_s) if tl_worker is not None else 0.0
                    lane_updates_hz = (float(lane_update_count) / elapsed_s) if lane_worker is not None else 0.0

                    print(
                        "[PROFILE] "
                        + f"avg_fps={avg_fps:.2f} avg_loop_ms={avg_loop_ms:.2f} | "
                        + f"capture={(profile_totals['capture'] / divisor) * 1000.0:.2f}ms "
                        + f"sign={(profile_totals['sign_infer'] / divisor) * 1000.0:.2f}ms "
                        + (f"tl={(profile_totals['tl_infer'] / divisor) * 1000.0:.2f}ms " if has_tl else "")
                        + (f"lane={(profile_totals['lane_infer'] / divisor) * 1000.0:.2f}ms " if has_lane else "")
                        + f"cons_sign={(profile_totals['sign_consensus'] / divisor) * 1000.0:.2f}ms "
                        + (f"cons_tl={(profile_totals['tl_consensus'] / divisor) * 1000.0:.2f}ms " if has_tl else "")
                        + (f"intersection={(profile_totals['intersection'] / divisor) * 1000.0:.2f}ms " if has_lane else "")
                        + f"draw={(profile_totals['draw'] / divisor) * 1000.0:.2f}ms "
                        + f"display={(profile_totals['display'] / divisor) * 1000.0:.2f}ms"
                        + (f" | stale(sign/tl/lane)={sign_stale_pct:.1f}/{tl_stale_pct:.1f}/{lane_stale_pct:.1f}%" if not args.sync_inference else "")
                        + (f" | capture_reuse={capture_reuse_pct:.1f}%" if args.async_capture else "")
                        + (f" | updates_hz(sign/tl/lane)={sign_updates_hz:.1f}/{tl_updates_hz:.1f}/{lane_updates_hz:.1f}" if not args.sync_inference else "")
                    )

            if key & 0xFF == ord("q"):
                break
    finally:
        if capture_stop_event is not None:
            capture_stop_event.set()
        if capture_thread is not None:
            capture_thread.join(timeout=1.0)
        if sign_worker is not None:
            sign_worker.shutdown(wait=False, cancel_futures=True)
        if tl_worker is not None:
            tl_worker.shutdown(wait=False, cancel_futures=True)
        if lane_worker is not None:
            lane_worker.shutdown(wait=False, cancel_futures=True)
        if speed_telemetry_reader is not None:
            speed_telemetry_reader.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Inference — single image
# ---------------------------------------------------------------------------

def run_inference_image(args, detector, tl_detector=None, lane_model=None):
    if not os.path.exists(args.image):
        print(f"Error: Target image file {args.image} does not exist.")
        return

    has_tl = tl_detector is not None
    has_lane = lane_model is not None
    tl_conf = args.tl_conf_threshold if args.tl_conf_threshold is not None else args.conf_threshold
    lane_conf = args.lane_conf_threshold if args.lane_conf_threshold is not None else args.conf_threshold
    lane_imgsz = args.lane_imgsz if args.lane_imgsz is not None else args.imgsz
    lane_scale = args.lane_inference_scale if args.lane_inference_scale is not None else args.inference_scale

    print(f"Running inference on image: {args.image}")
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image at {args.image}")
        return

    sign_detections, _ = run_detection_once(detector, frame, args.conf_threshold, args.imgsz, args.inference_scale)
    for (x1, y1, x2, y2, conf, cls) in sign_detections:
        class_name = resolve_class_name(detector, cls)
        _draw_detection(frame, x1, y1, x2, y2, f"{class_name}: {conf:.2f}", _SIGN_BOX_COLOR, _SIGN_TEXT_COLOR)

    if has_tl:
        tl_detections, _ = run_detection_once(tl_detector, frame, tl_conf, args.imgsz, args.inference_scale)
        for (x1, y1, x2, y2, conf, cls) in tl_detections:
            class_name = resolve_class_name(tl_detector, cls)
            color = _tl_color(class_name)
            _draw_detection(frame, x1, y1, x2, y2, f"TL-{class_name}: {conf:.2f}", color, color)

    if has_lane:
        lane_instances, _ = run_lane_segmentation_once(lane_model, frame, lane_conf, lane_imgsz, lane_scale)
        draw_lane_instances(frame, lane_instances, alpha=args.lane_mask_alpha, draw_contours=args.lane_draw_contours)
        score, is_intersection, reasons = compute_intersection_status(
            lane_instances,
            frame.shape,
            threshold=args.intersection_threshold,
        )
        text = f"Intersection: {'YES' if is_intersection else 'no'} ({score:.2f})"
        if reasons:
            text += " | " + ",".join(reasons)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 220) if is_intersection else (80, 180, 80), 2)

    _draw_legend(frame, has_tl, has_lane)

    window_name = f"{args.model} - Image Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)
    cv2.imshow(window_name, resize_to_fit(frame, args.window_width, args.window_height))
    print("Press any key inside the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.inference_scale <= 0.0:
        print(f"[WARNING] Invalid --inference-scale={args.inference_scale}. Falling back to 1.0.")
        args.inference_scale = 1.0
    elif args.inference_scale > 1.0:
        print(f"[WARNING] --inference-scale={args.inference_scale} is above 1.0. Clamping to 1.0.")
        args.inference_scale = 1.0

    if args.lane_inference_scale is not None:
        if args.lane_inference_scale <= 0.0:
            print(f"[WARNING] Invalid --lane-inference-scale={args.lane_inference_scale}. Falling back to --inference-scale.")
            args.lane_inference_scale = None
        elif args.lane_inference_scale > 1.0:
            print(f"[WARNING] --lane-inference-scale={args.lane_inference_scale} is above 1.0. Clamping to 1.0.")
            args.lane_inference_scale = 1.0

    if args.min_confirm_frames < 1:
        print(f"[WARNING] Invalid --min-confirm-frames={args.min_confirm_frames}. Falling back to 1.")
        args.min_confirm_frames = 1

    if args.max_missing_frames < 0:
        print(f"[WARNING] Invalid --max-missing-frames={args.max_missing_frames}. Falling back to 0.")
        args.max_missing_frames = 0

    if args.track_iou_threshold <= 0.0:
        print(f"[WARNING] Invalid --track-iou-threshold={args.track_iou_threshold}. Falling back to 0.35.")
        args.track_iou_threshold = 0.35
    elif args.track_iou_threshold > 1.0:
        print(f"[WARNING] --track-iou-threshold={args.track_iou_threshold} is above 1.0. Clamping to 1.0.")
        args.track_iou_threshold = 1.0

    if args.lane_mask_alpha < 0.0:
        print(f"[WARNING] Invalid --lane-mask-alpha={args.lane_mask_alpha}. Falling back to 0.0.")
        args.lane_mask_alpha = 0.0
    elif args.lane_mask_alpha > 1.0:
        print(f"[WARNING] --lane-mask-alpha={args.lane_mask_alpha} is above 1.0. Clamping to 1.0.")
        args.lane_mask_alpha = 1.0

    if args.intersection_threshold < 0.0:
        print(f"[WARNING] Invalid --intersection-threshold={args.intersection_threshold}. Falling back to 0.0.")
        args.intersection_threshold = 0.0
    elif args.intersection_threshold > 1.0:
        print(f"[WARNING] --intersection-threshold={args.intersection_threshold} is above 1.0. Clamping to 1.0.")
        args.intersection_threshold = 1.0

    if args.profile_every < 1:
        print(f"[WARNING] Invalid --profile-every={args.profile_every}. Falling back to 1.")
        args.profile_every = 1

    try:
        print(f"Initializing {args.model.upper()}...")
        detector = ModelFactory.create(
            model_name=args.model,
            model_path=args.weights,
            device=args.device,
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

    # Optionally load the traffic-light model
    tl_detector = None
    if args.tl_weights:
        try:
            print(f"Initializing traffic-light model from {args.tl_weights}...")
            tl_detector = ModelFactory.create(
                model_name=args.model,   # always YOLO
                model_path=args.tl_weights,
                device=args.device,
            )
            print("[INFO] Traffic-light model loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load traffic-light model: {e}. Continuing with sign model only.")
            tl_detector = None

    lane_model = None
    if args.lane_weights:
        try:
            print(f"Initializing lane segmentation model from {args.lane_weights}...")
            lane_model = YOLO(args.lane_weights)
            lane_model.to(args.device)
            print("[INFO] Lane segmentation model loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load lane model: {e}. Continuing without lane overlays.")
            lane_model = None

    if args.mode == "train":
        run_training(args, detector)
    elif args.mode == "test":
        run_evaluation(args, detector)
    elif args.mode == "inference":
        if args.weights is None:
            print("WARNING: Inference is being run without weights (--weights). Accuracy might be 0.")
        if args.image:
            run_inference_image(args, detector, tl_detector, lane_model)
        else:
            run_inference_screen_capture(args, detector, tl_detector, lane_model)


if __name__ == "__main__":
    main()