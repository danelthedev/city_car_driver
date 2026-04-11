import os
import shutil
import sys

from ultralytics import YOLO

from args import parse_args
from inference import run_inference_screen_capture, run_inference_image
from perception.traffic_signs.factory import ModelFactory


def run_training(args, detector):
    print(f"Starting training for {args.model} on {args.data} for {args.epochs} epochs.")
    detector.train(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)

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


def _validate_args(args):
    if args.inference_scale <= 0.0 or args.inference_scale > 1.0:
        print(f"[WARNING] --inference-scale={args.inference_scale} out of range. Clamping to (0, 1].")
        args.inference_scale = max(0.01, min(1.0, args.inference_scale))

    if args.lane_inference_scale is not None:
        if args.lane_inference_scale <= 0.0 or args.lane_inference_scale > 1.0:
            print(f"[WARNING] --lane-inference-scale={args.lane_inference_scale} out of range. Falling back to --inference-scale.")
            args.lane_inference_scale = None

    if args.min_confirm_frames < 1:
        print("[WARNING] --min-confirm-frames must be >= 1. Falling back to 1.")
        args.min_confirm_frames = 1

    if args.max_missing_frames < 0:
        print("[WARNING] --max-missing-frames must be >= 0. Falling back to 0.")
        args.max_missing_frames = 0

    if not (0.0 < args.track_iou_threshold <= 1.0):
        print(f"[WARNING] --track-iou-threshold={args.track_iou_threshold} out of range. Clamping to (0, 1].")
        args.track_iou_threshold = max(1e-3, min(1.0, args.track_iou_threshold))

    args.lane_mask_alpha = max(0.0, min(1.0, args.lane_mask_alpha))
    args.intersection_threshold = max(0.0, min(1.0, args.intersection_threshold))


def main():
    args = parse_args()
    _validate_args(args)

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

    tl_detector = None
    if args.tl_weights:
        try:
            print(f"Initializing traffic-light model from {args.tl_weights}...")
            tl_detector = ModelFactory.create(
                model_name=args.model,
                model_path=args.tl_weights,
                device=args.device,
            )
            print("[INFO] Traffic-light model loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load traffic-light model: {e}. Continuing without.")

    lane_model = None
    if args.lane_weights:
        try:
            print(f"Initializing lane segmentation model from {args.lane_weights}...")
            lane_model = YOLO(args.lane_weights)
            lane_model.to(args.device)
            print("[INFO] Lane segmentation model loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load lane model: {e}. Continuing without.")

    if args.mode == "train":
        run_training(args, detector)
    elif args.mode == "test":
        run_evaluation(args, detector)
    elif args.mode == "inference":
        if args.weights is None:
            print("WARNING: Inference running without --weights. Accuracy may be 0.")
        if args.image:
            run_inference_image(args, detector, tl_detector, lane_model)
        else:
            run_inference_screen_capture(args, detector, tl_detector, lane_model)


if __name__ == "__main__":
    main()