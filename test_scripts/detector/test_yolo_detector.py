"""
Detection script for YOLO Traffic Sign Detector.
This script demonstrates how to use the trained model to detect traffic signs in images.
"""

from modules.traffic_signs.yolo_detector import YoloTrafficSignDetector
from pathlib import Path
import cv2
import argparse

def detect_from_image(model_path: str, image_path: str, output_path: str = None,
                     confidence_threshold: float = 0.25, show: bool = False,
                     img_size: int = 416, use_half: bool = True, save_crops: bool = False,
                     multi_scale: bool = False, scales: str = '1.0', tta: bool = False,
                     adaptive_conf: bool = False, min_conf_small: float = 0.10, small_area_ratio: float = 0.002,
                     agnostic_nms: bool = False, max_det: int = 300):
    """
    Detect traffic signs in a single image.

    Args:
        model_path: Path to the trained YOLO model
        image_path: Path to the input image
        output_path: Path to save the annotated image (optional)
        confidence_threshold: Minimum confidence for detections (default: 0.25 for better recall)
        show: Whether to display the result
        img_size: Base image size for inference (default: 640)
        use_half: Use FP16 if on GPU (default: True)
        save_crops: Save cropped detections to tmp/ (default: False)
        multi_scale: Enable multi-scale inference (default: False)
        scales: Comma-separated scales for multi-scale (default: 1.0,1.25)
        tta: Enable test-time augmentation (default: False)
        adaptive_conf: Enable adaptive confidence for small objects (default: False)
        min_conf_small: Minimum confidence for small objects (default: 0.10)
        small_area_ratio: Area ratio threshold for small objects (default: 0.002)
        agnostic_nms: Use class-agnostic NMS (default: False)
        max_det: Max detections per image (default: 300)
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

    # Load the detector with trained model and optimized settings
    print(f"\nLoading model from: {model_path}")
    # Parse scales string into tuple
    scale_vals = tuple(float(s) for s in scales.split(',') if s.strip())
    detector = YoloTrafficSignDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        img_size=img_size,
        use_half=use_half,
        save_crops=save_crops,
        multi_scale=multi_scale,
        scales=scale_vals,
        tta=tta,
        adaptive_conf=adaptive_conf,
        min_conf_small=min_conf_small,
        small_area_ratio=small_area_ratio,
        agnostic_nms=agnostic_nms,
        max_det=max_det
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
        "--img-size",
        type=int,
        default=640,
        help="Base image size for inference (default: 640)"
    )

    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable multi-scale inference"
    )

    parser.add_argument(
        "--scales",
        type=str,
        default="1.0,1.25",
        help="Comma-separated scales for multi-scale (default: 1.0,1.25)"
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation"
    )

    parser.add_argument(
        "--adaptive-conf",
        action="store_true",
        help="Enable adaptive confidence for small objects"
    )

    parser.add_argument(
        "--min-conf-small",
        type=float,
        default=0.10,
        help="Minimum confidence for small objects"
    )

    parser.add_argument(
        "--small-area-ratio",
        type=float,
        default=0.002,
        help="Area ratio threshold for small objects"
    )

    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS"
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per image"
    )

    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped detections to tmp/"
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
        img_size=args.img_size,
        use_half=True,
        save_crops=args.save_crops,
        multi_scale=args.multi_scale,
        scales=args.scales,
        tta=args.tta,
        adaptive_conf=args.adaptive_conf,
        min_conf_small=args.min_conf_small,
        small_area_ratio=args.small_area_ratio,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det
    )

if __name__ == "__main__":
    main()
