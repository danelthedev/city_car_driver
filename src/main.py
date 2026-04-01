import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import sys
import time

import cv2
import mss
import numpy as np

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
        "--sync-inference",
        action="store_true",
        help="Run inference synchronously. Default behavior uses async inference for smoother display.",
    )
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=1,
        help="Monitor index for screen capture (mss indexing: 1..N).",
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


def _tl_color(class_name: str):
    return _TL_CLASS_COLORS.get(class_name.lower(), _TL_DEFAULT_COLOR)


def _draw_detection(frame, x1, y1, x2, y2, label, box_color, text_color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(
        frame, label, (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2,
    )


def _draw_legend(frame, has_tl_model: bool):
    """Draw a small legend in the top-right corner."""
    items = [("Signs", _SIGN_BOX_COLOR)]
    if has_tl_model:
        items += [
            ("TL: red",    _TL_CLASS_COLORS["red"]),
            ("TL: yellow", _TL_CLASS_COLORS["yellow"]),
            ("TL: green",  _TL_CLASS_COLORS["green"]),
            ("TL: off",    _TL_CLASS_COLORS["off"]),
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

def run_inference_screen_capture(args, detector, tl_detector=None):
    print("Starting screen capture inference...")
    print("Press 'q' in the window to stop.")

    has_tl = tl_detector is not None
    tl_conf = args.tl_conf_threshold if args.tl_conf_threshold is not None else args.conf_threshold

    inference_mode = "sync" if args.sync_inference else "async"
    print(
        f"Inference settings: conf={args.conf_threshold:.2f}, "
        + (f"tl_conf={tl_conf:.2f}, " if has_tl else "")
        + f"imgsz={args.imgsz}, infer_scale={args.inference_scale:.2f}, "
        f"mode={inference_mode}, confirm={args.min_confirm_frames}, "
        f"max_miss={args.max_missing_frames}, track_iou={args.track_iou_threshold:.2f}, "
        f"window={args.window_width}x{args.window_height}"
    )
    if has_tl:
        print("[INFO] Dual-model mode: sign model + traffic-light model running in parallel.")

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
    sign_tracks = {}
    sign_next_id = 1
    sign_raw_count = 0
    sign_infer_duration = 0.0

    tl_detections_raw = []
    tl_stable = []
    tl_tracks = {}
    tl_next_id = 1
    tl_raw_count = 0
    tl_infer_duration = 0.0

    display_fps_ema = 0.0
    sign_fps_ema = 0.0
    tl_fps_ema = 0.0

    # Each model gets its own single-thread executor for async mode
    sign_worker = None
    tl_worker = None
    sign_pending = None
    tl_pending = None

    if not args.sync_inference:
        sign_worker = ThreadPoolExecutor(max_workers=1)
        if has_tl:
            tl_worker = ThreadPoolExecutor(max_workers=1)

    try:
        while True:
            loop_start = time.perf_counter()

            screen_img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

            # ----------------------------------------------------------
            # Sign model
            # ----------------------------------------------------------
            if sign_worker is None:
                sign_detections_raw, sign_infer_duration = run_detection_once(
                    detector, frame, args.conf_threshold, args.imgsz, args.inference_scale
                )
                sign_raw_count = len(sign_detections_raw)
                sign_stable, sign_next_id = apply_temporal_consensus(
                    sign_detections_raw, sign_tracks, sign_next_id,
                    args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                )
            else:
                if sign_pending is not None and sign_pending.done():
                    try:
                        sign_detections_raw, sign_infer_duration = sign_pending.result()
                        sign_raw_count = len(sign_detections_raw)
                        sign_stable, sign_next_id = apply_temporal_consensus(
                            sign_detections_raw, sign_tracks, sign_next_id,
                            args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                        )
                    except Exception as exc:
                        print(f"[ERROR] Sign model inference failed: {exc}")
                        sign_detections_raw = []
                        sign_stable = []
                        sign_tracks.clear()
                    sign_pending = None

                if sign_pending is None:
                    sign_pending = sign_worker.submit(
                        run_detection_once,
                        detector, frame.copy(), args.conf_threshold, args.imgsz, args.inference_scale,
                    )

            # ----------------------------------------------------------
            # Traffic light model (if provided)
            # ----------------------------------------------------------
            if has_tl:
                if tl_worker is None:
                    tl_detections_raw, tl_infer_duration = run_detection_once(
                        tl_detector, frame, tl_conf, args.imgsz, args.inference_scale
                    )
                    tl_raw_count = len(tl_detections_raw)
                    tl_stable, tl_next_id = apply_temporal_consensus(
                        tl_detections_raw, tl_tracks, tl_next_id,
                        args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                    )
                else:
                    if tl_pending is not None and tl_pending.done():
                        try:
                            tl_detections_raw, tl_infer_duration = tl_pending.result()
                            tl_raw_count = len(tl_detections_raw)
                            tl_stable, tl_next_id = apply_temporal_consensus(
                                tl_detections_raw, tl_tracks, tl_next_id,
                                args.min_confirm_frames, args.max_missing_frames, args.track_iou_threshold,
                            )
                        except Exception as exc:
                            print(f"[ERROR] Traffic light model inference failed: {exc}")
                            tl_detections_raw = []
                            tl_stable = []
                            tl_tracks.clear()
                        tl_pending = None

                    if tl_pending is None:
                        tl_pending = tl_worker.submit(
                            run_detection_once,
                            tl_detector, frame.copy(), tl_conf, args.imgsz, args.inference_scale,
                        )

            # ----------------------------------------------------------
            # Draw sign detections (green)
            # ----------------------------------------------------------
            sign_to_draw = sign_stable
            if args.max_draw_detections > 0:
                sign_to_draw = sign_to_draw[: args.max_draw_detections]

            for (x1, y1, x2, y2, conf, cls) in sign_to_draw:
                class_name = resolve_class_name(detector, cls)
                _draw_detection(
                    frame, x1, y1, x2, y2,
                    f"{class_name}: {conf:.2f}",
                    _SIGN_BOX_COLOR, _SIGN_TEXT_COLOR,
                )

            # ----------------------------------------------------------
            # Draw traffic light detections (colour-coded)
            # ----------------------------------------------------------
            if has_tl:
                tl_to_draw = tl_stable
                if args.max_draw_detections > 0:
                    tl_to_draw = tl_to_draw[: args.max_draw_detections]

                for (x1, y1, x2, y2, conf, cls) in tl_to_draw:
                    class_name = resolve_class_name(tl_detector, cls)
                    color = _tl_color(class_name)
                    _draw_detection(
                        frame, x1, y1, x2, y2,
                        f"TL-{class_name}: {conf:.2f}",
                        color, color,
                    )

            # ----------------------------------------------------------
            # HUD
            # ----------------------------------------------------------
            loop_duration = max(time.perf_counter() - loop_start, 1e-6)
            display_fps = 1.0 / loop_duration
            sign_fps = 1.0 / sign_infer_duration if sign_infer_duration > 0.0 else 0.0
            tl_fps = 1.0 / tl_infer_duration if tl_infer_duration > 0.0 else 0.0

            display_fps_ema = display_fps if display_fps_ema == 0.0 else 0.9 * display_fps_ema + 0.1 * display_fps
            if sign_fps > 0.0:
                sign_fps_ema = sign_fps if sign_fps_ema == 0.0 else 0.9 * sign_fps_ema + 0.1 * sign_fps
            if tl_fps > 0.0:
                tl_fps_ema = tl_fps if tl_fps_ema == 0.0 else 0.9 * tl_fps_ema + 0.1 * tl_fps

            mode_tag = "SYNC" if args.sync_inference else "ASYNC"
            cv2.putText(frame, f"Display FPS: {display_fps_ema:.1f} | Sign FPS: {sign_fps_ema:.1f}" + (f" | TL FPS: {tl_fps_ema:.1f}" if has_tl else ""), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            cv2.putText(frame, f"Mode: {mode_tag} | Scale: {args.inference_scale:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Signs raw/stable: {sign_raw_count}/{len(sign_to_draw)}" + (f"  |  TL raw/stable: {tl_raw_count}/{len(tl_to_draw) if has_tl else 0}" if has_tl else ""), (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            _draw_legend(frame, has_tl)

            disp_frame = resize_to_fit(frame, args.window_width, args.window_height)
            cv2.imshow(window_name, disp_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if sign_worker is not None:
            sign_worker.shutdown(wait=False, cancel_futures=True)
        if tl_worker is not None:
            tl_worker.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Inference — single image
# ---------------------------------------------------------------------------

def run_inference_image(args, detector, tl_detector=None):
    if not os.path.exists(args.image):
        print(f"Error: Target image file {args.image} does not exist.")
        return

    has_tl = tl_detector is not None
    tl_conf = args.tl_conf_threshold if args.tl_conf_threshold is not None else args.conf_threshold

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

    _draw_legend(frame, has_tl)

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

    if args.mode == "train":
        run_training(args, detector)
    elif args.mode == "test":
        run_evaluation(args, detector)
    elif args.mode == "inference":
        if args.weights is None:
            print("WARNING: Inference is being run without weights (--weights). Accuracy might be 0.")
        if args.image:
            run_inference_image(args, detector, tl_detector)
        else:
            run_inference_screen_capture(args, detector, tl_detector)


if __name__ == "__main__":
    main()