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


def run_inference_screen_capture(args, detector):
    print("Starting screen capture inference...")
    print("Press 'q' in the window to stop.")
    print(f"Inference settings: conf={args.conf_threshold:.2f}, imgsz={args.imgsz}")

    sct = mss.mss()
    # Typically, city car driving runs on the main monitor. 
    # Grab the dimensions of the primary monitor.
    monitor = sct.monitors[1] 
    
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
            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # 5. Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 6. Show the frame tracking
        # Resize to fit on screen if monitor is huge
        disp_frame = cv2.resize(frame, (800, 600))
        cv2.imshow(f"{args.model} - Real-time Inference", disp_frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_inference_image(args, detector):
    if not os.path.exists(args.image):
        print(f"Error: Target image file {args.image} does not exist.")
        return
        
    print(f"Running inference on image: {args.image}")
    print(f"Inference settings: conf={args.conf_threshold:.2f}, imgsz={args.imgsz}")
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
        label = f"Class {cls}: {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Resize and display
    h, w = frame.shape[:2]
    # Simple logic to ensure massive images fit on screen
    if h > 900 or w > 1600:
        scale = min(1600/w, 900/h)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
    cv2.imshow(f"{args.model} - Image Inference", frame)
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