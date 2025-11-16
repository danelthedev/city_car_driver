"""
Detection script for YOLO Traffic Sign Detector.
This script demonstrates how to use the trained model to detect traffic signs in images.
"""

from modules.traffic_signs.yolo_detector import YoloTrafficSignDetector
from pathlib import Path
import cv2
import argparse
from datetime import datetime
import time

def detect_from_image(model_path: str, image_path: str, output_path: str = None,
                     confidence_threshold: float = 0.25, show: bool = False,
                     save_cutouts: bool = True, cutouts_dir: str = "tmp"):
    print("=" * 60)
    print("YOLO Traffic Sign Detector - Detection")
    print("=" * 60)

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train_yolo_detector.py")
        return

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found at {image_path}")
        return

    # Load the detector with trained model and optimized settings
    print(f"\nLoading model from: {model_path}")
    detector = YoloTrafficSignDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )

    # Load the image
    print(f"Loading image from: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return

    print(f"Image shape: {image.shape}")

    # Detect traffic signs
    print("\nDetecting traffic signs...")
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = time.time() - start_time

    print(f"\nFound {len(detections)} traffic sign(s):")
    print(f"Inference time: {inference_time:.3f} seconds ({inference_time*1000:.1f} ms)")
    print("-" * 60)

    for i, det in enumerate(detections, 1):
        bbox = det['bbox']
        confidence = det['confidence']

        print(f"\nDetection {i}:")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

    # Save cutouts of detected signs
    if len(detections) > 0 and save_cutouts:
        print("\n" + "-" * 60)
        print("Saving traffic sign cutouts...")

        # Create cutouts directory
        cutouts_path = Path(cutouts_dir)
        cutouts_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem

        saved_cutouts = []
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            confidence = det['confidence']

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)

            # Add padding to cutout (10% of width/height)
            h, w = image.shape[:2]
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            padding_x = int(bbox_w * 0.1)
            padding_y = int(bbox_h * 0.1)

            # Apply padding with boundary checks
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(w, x2 + padding_x)
            y2_padded = min(h, y2 + padding_y)

            # Extract cutout
            cutout = image[y1_padded:y2_padded, x1_padded:x2_padded]

            # Generate filename
            cutout_filename = f"{image_name}_{timestamp}_sign{i:02d}_conf{confidence:.2f}.jpg"
            cutout_path = cutouts_path / cutout_filename

            # Save cutout
            cv2.imwrite(str(cutout_path), cutout)
            saved_cutouts.append(str(cutout_path))
            print(f"  Saved cutout {i}: {cutout_path}")

        print(f"\nTotal {len(saved_cutouts)} cutout(s) saved to: {cutouts_path}")


    # Visualize detections
    if len(detections) > 0:
        print("\n" + "-" * 60)
        print("Visualizing detections...")

        annotated_image = detector.visualize_detections(
            image=image,
            detections=detections,
            save_path=output_path,
            show=show
        )

    else:
        print("\nNo traffic signs detected in the image.")

    # Display performance metrics
    fps = 1.0 / inference_time if inference_time > 0 else 0
    print("\n" + "-" * 60)
    print("Performance Metrics:")
    print(f"  Inference Time: {inference_time:.3f} seconds")
    print(f"  Milliseconds: {inference_time*1000:.1f} ms")
    print(f"  FPS: {fps:.2f}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Detection Completed!")
    print("=" * 60)

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Detect traffic signs in images using trained YOLO model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo/best.pt",
        help="Path to the trained YOLO model (default: models/yolo/best.pt)"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the annotated image (default: None)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25 for better recall)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image"
    )

    parser.add_argument(
        "--save-cutouts",
        action="store_true",
        default=True,
        help="Save cutouts of detected traffic signs (default: True)"
    )

    parser.add_argument(
        "--no-cutouts",
        action="store_true",
        help="Disable saving cutouts of detected traffic signs"
    )

    parser.add_argument(
        "--cutouts-dir",
        type=str,
        default="tmp",
        help="Directory to save traffic sign cutouts (default: tmp)"
    )

    args = parser.parse_args()

    # If no output path is specified, create one based on input
    if args.output is None:
        input_path = Path(args.image)
        args.output = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")

    detect_from_image(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        confidence_threshold=args.confidence,
        show=args.show,
        save_cutouts=not args.no_cutouts,
        cutouts_dir=args.cutouts_dir
    )

if __name__ == "__main__":
    main()
