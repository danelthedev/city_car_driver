import argparse
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


def run_inference_screen_capture(args, detector):
    print("Starting screen capture inference...")
    print("Press 'q' in the window to stop.")
    print(
        f"Inference settings: conf={args.conf_threshold:.2f}, imgsz={args.imgsz}, "
        f"window={args.window_width}x{args.window_height}"
    )

    sct = mss.mss()
    # Typically, city car driving runs on the main monitor. 
    # Grab the dimensions of the primary monitor.
    monitor = sct.monitors[1]
    window_name = f"{args.model} - Real-time Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)
    
    while True:
        start_time = time.time()
        
        # 1. Grab screen
        screen_img = np.array(sct.grab(monitor))
        
        # 2. Convert from BGRA (mss default) to BGR (OpenCV/ultralytics expected)
        frame = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
        
        # 3. Predict mapping confidence and box
        detections = detector.predict(
            frame,
            confidence_threshold=args.conf_threshold,
            image_size=args.imgsz,
        )
        
        # 4. Draw detections on the frame
        for (x1, y1, x2, y2, conf, cls) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = resolve_class_name(detector, cls)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # 5. Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 6. Show the frame tracking
        disp_frame = resize_to_fit(frame, args.window_width, args.window_height)
        cv2.imshow(window_name, disp_frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_inference_image(args, detector):
    if not os.path.exists(args.image):
        print(f"Error: Target image file {args.image} does not exist.")
        return
        
    print(f"Running inference on image: {args.image}")
    print(
        f"Inference settings: conf={args.conf_threshold:.2f}, imgsz={args.imgsz}, "
        f"window={args.window_width}x{args.window_height}"
    )
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image at {args.image}")
        return

    # Predict
    detections = detector.predict(
        frame,
        confidence_threshold=args.conf_threshold,
        image_size=args.imgsz,
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