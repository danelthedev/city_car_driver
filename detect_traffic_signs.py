"""
Detection script for YOLO Traffic Sign Detector.
This script demonstrates how to use the trained model to detect traffic signs in images.
"""

from modules.traffic_signs.yolo_detector import YoloTrafficSignDetector
from pathlib import Path
import cv2
import argparse

def detect_from_image(model_path: str, image_path: str, output_path: str = None,
                     confidence_threshold: float = 0.5, show: bool = False):
    """
    Detect traffic signs in a single image.

    Args:
        model_path: Path to the trained YOLO model
        image_path: Path to the input image
        output_path: Path to save the annotated image (optional)
        confidence_threshold: Minimum confidence for detections
        show: Whether to display the result
    """
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

    # Load the detector with trained model
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
    detections = detector.detect(image)

    print(f"\nFound {len(detections)} traffic sign(s):")
    print("-" * 60)

    for i, det in enumerate(detections, 1):
        bbox = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']

        print(f"\nDetection {i}:")
        print(f"  Class: {class_name} (ID: {class_id})")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

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

        if output_path:
            print(f"Annotated image saved to: {output_path}")
    else:
        print("\nNo traffic signs detected in the image.")

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
        default="models/gtsdb_best.pt",
        help="Path to the trained YOLO model (default: models/gtsdb_best.pt)"
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
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image"
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
        show=args.show
    )

if __name__ == "__main__":
    main()

