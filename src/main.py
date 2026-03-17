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
        help="Mode to run the application in."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["yolo", "fasterrcnn", "ssd"],
        help="The detection model string identifier."
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/traffic_sign_dataset/dataset.yaml",
        help="Path to the dataset config or root folder for train/test modes."
    )
    
    parser.add_argument(
        "--weights", 
        type=str, 
        default=None,
        help="Path to pre-trained model weights. E.g. yolov8n.pt, yolo11m.pt. Required for evaluation or inference."
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run the model on (cuda or cpu)."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--image", type=str, default=None, help="Path to image file for inference (instead of screen capture).")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.20,
        help="Minimum confidence used during inference.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size. Higher values improve small-sign recall but reduce FPS.",
    )
    parser.add_argument(
        "--inference-scale",
        type=float,
        default=1.0,
        help="Scale applied before inference only (0 < value <= 1). Lower values increase FPS with some accuracy loss.",
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


def run_training(args, detector):
    print(f"Starting training for {args.model} on {args.data} for {args.epochs} epochs.")
    
    # Run the underlying training method
    results = detector.train(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)
    
    # YOLO specific: Auto-copy the best weights to the top-level 'models/' dir
    if args.model == "yolo" and hasattr(detector.model.trainer, 'best'):
        best_weights_path = detector.model.trainer.best
        if best_weights_path and os.path.exists(best_weights_path):
            os.makedirs("models", exist_ok=True)
            # Use the base weights filename to determine target file (e.g. yolo11m_best.pt)
            base_model_name = os.path.splitext(os.path.basename(args.weights or "yolov8n"))[0]
            target_path = os.path.join("models", f"{base_model_name}_best.pt")
            shutil.copy2(best_weights_path, target_path)
            print(f"\n[INFO] Best {base_model_name} model copied to: {target_path}")
        else:
            print("\n[WARNING] Could not locate best.pt to copy to models directory.")


def run_evaluation(args, detector):
    if not args.weights:
        print("Warning: No weights provided for evaluation. Using base model randomly initialized or pretrained.")
    print(f"Starting evaluation for {args.model} on {args.data}.")
    metrics = detector.evaluate(data_path=args.data)
    print("Metrics:", metrics)


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


def scale_detections_to_original(
    detections,
    scale: float,
    frame_width: int,
    frame_height: int,
):
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


def run_detection_once(
    detector,
    frame: np.ndarray,
    confidence_threshold: float,
    image_size: int,
    inference_scale: float,
):
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


def run_inference_screen_capture(args, detector):
    print("Starting screen capture inference...")
    print("Press 'q' in the window to stop.")
    inference_mode = "sync" if args.sync_inference else "async"
    print(
        f"Inference settings: conf={args.conf_threshold:.2f}, imgsz={args.imgsz}, "
        f"infer_scale={args.inference_scale:.2f}, mode={inference_mode}, "
        f"confirm={args.min_confirm_frames}, max_miss={args.max_missing_frames}, "
        f"track_iou={args.track_iou_threshold:.2f}, "
        f"window={args.window_width}x{args.window_height}"
    )

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

    latest_detections = []
    stable_detections = []
    last_inference_duration = 0.0
    display_fps_ema = 0.0
    inference_fps_ema = 0.0
    raw_detection_count = 0

    tracks = {}
    next_track_id = 1

    inference_worker = None
    pending_inference = None
    if not args.sync_inference:
        inference_worker = ThreadPoolExecutor(max_workers=1)
    
    try:
        while True:
            loop_start = time.perf_counter()

            screen_img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

            if inference_worker is None:
                latest_detections, last_inference_duration = run_detection_once(
                    detector,
                    frame,
                    args.conf_threshold,
                    args.imgsz,
                    args.inference_scale,
                )
                raw_detection_count = len(latest_detections)
                stable_detections, next_track_id = apply_temporal_consensus(
                    latest_detections,
                    tracks,
                    next_track_id,
                    args.min_confirm_frames,
                    args.max_missing_frames,
                    args.track_iou_threshold,
                )
            else:
                if pending_inference is not None and pending_inference.done():
                    try:
                        latest_detections, last_inference_duration = pending_inference.result()
                        raw_detection_count = len(latest_detections)
                        stable_detections, next_track_id = apply_temporal_consensus(
                            latest_detections,
                            tracks,
                            next_track_id,
                            args.min_confirm_frames,
                            args.max_missing_frames,
                            args.track_iou_threshold,
                        )
                    except Exception as exc:
                        print(f"[ERROR] Async inference failed: {exc}")
                        latest_detections = []
                        stable_detections = []
                        last_inference_duration = 0.0
                        raw_detection_count = 0
                        tracks.clear()
                    pending_inference = None

                if pending_inference is None:
                    pending_inference = inference_worker.submit(
                        run_detection_once,
                        detector,
                        frame.copy(),
                        args.conf_threshold,
                        args.imgsz,
                        args.inference_scale,
                    )

            detections_to_draw = stable_detections
            if args.max_draw_detections > 0 and len(detections_to_draw) > args.max_draw_detections:
                detections_to_draw = detections_to_draw[:args.max_draw_detections]

            for (x1, y1, x2, y2, conf, cls) in detections_to_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_name = resolve_class_name(detector, cls)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            loop_duration = max(time.perf_counter() - loop_start, 1e-6)
            display_fps = 1.0 / loop_duration
            inference_fps = 1.0 / last_inference_duration if last_inference_duration > 0.0 else 0.0

            if display_fps_ema == 0.0:
                display_fps_ema = display_fps
            else:
                display_fps_ema = (0.90 * display_fps_ema) + (0.10 * display_fps)

            if inference_fps > 0.0:
                if inference_fps_ema == 0.0:
                    inference_fps_ema = inference_fps
                else:
                    inference_fps_ema = (0.90 * inference_fps_ema) + (0.10 * inference_fps)

            mode_tag = "SYNC" if args.sync_inference else "ASYNC"
            cv2.putText(
                frame,
                f"Display FPS: {display_fps_ema:.1f} | Infer FPS: {inference_fps_ema:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Mode: {mode_tag} | Infer scale: {args.inference_scale:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Detections raw/stable: {raw_detection_count}/{len(detections_to_draw)}",
                (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

            disp_frame = resize_to_fit(frame, args.window_width, args.window_height)
            cv2.imshow(window_name, disp_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if inference_worker is not None:
            inference_worker.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


def run_inference_image(args, detector):
    if not os.path.exists(args.image):
        print(f"Error: Target image file {args.image} does not exist.")
        return
        
    print(f"Running inference on image: {args.image}")
    print(
        f"Inference settings: conf={args.conf_threshold:.2f}, imgsz={args.imgsz}, "
        f"infer_scale={args.inference_scale:.2f}, "
        f"window={args.window_width}x{args.window_height}"
    )
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image at {args.image}")
        return

    detections, _ = run_detection_once(
        detector,
        frame,
        args.conf_threshold,
        args.imgsz,
        args.inference_scale,
    )
    
    # Draw detections
    for (x1, y1, x2, y2, conf, cls) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_name = resolve_class_name(detector, cls)
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    window_name = f"{args.model} - Image Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)
    disp_frame = resize_to_fit(frame, args.window_width, args.window_height)

    cv2.imshow(window_name, disp_frame)
    print("Press any key inside the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
        print(
            f"[WARNING] Invalid --track-iou-threshold={args.track_iou_threshold}. "
            "Falling back to 0.35."
        )
        args.track_iou_threshold = 0.35
    elif args.track_iou_threshold > 1.0:
        print(f"[WARNING] --track-iou-threshold={args.track_iou_threshold} is above 1.0. Clamping to 1.0.")
        args.track_iou_threshold = 1.0
    
    try:
        print(f"Initializing {args.model.upper()}...")
        detector = ModelFactory.create(
            model_name=args.model, 
            model_path=args.weights, 
            device=args.device
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

    if args.mode == "train":
        run_training(args, detector)
    elif args.mode == "test":
        run_evaluation(args, detector)
    elif args.mode == "inference":
        if args.weights is None:
            print("WARNING: Inference is being run without weights (--weights). Accuracy might be 0.")
        if args.image:
            run_inference_image(args, detector)
        else:
            run_inference_screen_capture(args, detector)


if __name__ == "__main__":
    main()