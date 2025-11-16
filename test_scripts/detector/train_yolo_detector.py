"""
Training script for YOLO Traffic Sign Detector.
This script trains a YOLO11n or YOLO11s model to detect all traffic signs (single class)
using the GTSDB_YOLO_1_CLASS dataset.
"""

from modules.traffic_signs.yolo_detector import YoloTrafficSignDetector
from pathlib import Path
import argparse
import torch


def train_detector(
    data_yaml: str = "data/datasets/GTSDB_YOLO_1_CLASS/data.yaml",
    model_size: str = 'n',
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 1280,
    project: str = "models",
    name: str = "yolo_detector2",
    device: str = 'cuda',
    resume: bool = False,
    pretrained: bool = True
):
    """
    Train the YOLO traffic sign detector.

    Args:
        data_yaml: Path to data.yaml configuration file
        model_size: YOLO model size ('n' for nano, 's' for small)
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Input image size
        project: Project directory for saving results
        name: Experiment name
        device: Device to use ('cuda', 'cpu', or None for auto)
        resume: Resume training from last checkpoint
        pretrained: Use pretrained weights
    """
    print("=" * 80)
    print("YOLO Traffic Sign Detector - Training")
    print("=" * 80)

    # Check if dataset exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Error: Dataset configuration not found at {data_yaml}")
        print("Please ensure the GTSDB_YOLO_1_CLASS dataset is properly set up.")
        return

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nTraining Configuration:")
    print(f"  Model: YOLO11{model_size}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {imgsz}")
    print(f"  Device: {device}")
    print(f"  Output: {project}/{name}")
    print(f"  Resume: {resume}")
    print(f"  Pretrained: {pretrained}")
    print("-" * 80)

    # Initialize detector
    detector = YoloTrafficSignDetector(
        model_path=None,
        confidence_threshold=0.25,
        model_size=model_size
    )

    # Additional training parameters
    train_params = {
        'patience': 20,          # Early stopping patience
        'save': True,            # Save checkpoints
        'save_period': 10,       # Save every 10 epochs
        'plots': True,           # Generate training plots
        'val': True,             # Validate during training
        'pretrained': pretrained,
        'optimizer': 'auto',     # AdamW optimizer
        'verbose': True,
        'seed': 42,              # Reproducibility
        'deterministic': False,  # Faster training
        'single_cls': True,      # Single class detection
        'rect': False,           # Rectangular training
        'cos_lr': True,          # Cosine learning rate scheduler
        'close_mosaic': 10,      # Close mosaic augmentation in last 10 epochs
        'resume': resume,
        'amp': True,             # Automatic mixed precision
        'fraction': 1.0,         # Use 100% of data
        'profile': False,
        'freeze': None,          # Layers to freeze
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.01,            # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,             # Box loss gain
        'cls': 0.5,             # Class loss gain
        'dfl': 1.5,             # DFL loss gain
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,         # HSV-Hue augmentation
        'hsv_s': 0.7,           # HSV-Saturation augmentation
        'hsv_v': 0.4,           # HSV-Value augmentation
        'degrees': 0.0,         # Rotation augmentation
        'translate': 0.1,       # Translation augmentation
        'scale': 0.5,           # Scale augmentation
        'shear': 0.0,           # Shear augmentation
        'perspective': 0.0,     # Perspective augmentation
        'flipud': 0.0,          # Vertical flip probability
        'fliplr': 0.5,          # Horizontal flip probability
        'mosaic': 1.0,          # Mosaic augmentation probability
        'mixup': 0.0,           # Mixup augmentation probability
        'copy_paste': 0.0,      # Copy-paste augmentation
    }

    # Start training
    try:
        results = detector.train(
            data_yaml=str(data_yaml),
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            **train_params
        )

        print("\n" + "=" * 80)
        print("Training Completed Successfully!")
        print("=" * 80)
        print(f"\nBest model saved to: {results['best_model_path']}")

        # Evaluate on validation set
        print("\n" + "-" * 80)
        print("Evaluating on validation set...")
        print("-" * 80)

        metrics = detector.evaluate(data_yaml=str(data_yaml))

        print("\n" + "=" * 80)
        print("Training Summary:")
        print("=" * 80)
        print(f"  Best Model: {results['best_model_path']}")
        print(f"  mAP@50: {metrics['mAP50']:.4f}")
        print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print("=" * 80)

    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"Error during training: {e}")
        print("=" * 80)
        raise


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11 model for traffic sign detection (single class)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/datasets/GTSDB_YOLO_1_CLASS/data.yaml",
        help="Path to data.yaml configuration file"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default='n',
        choices=['n', 's'],
        help="YOLO model size (n=nano, s=small)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="models",
        help="Project directory for saving results"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo_detector",
        help="Experiment name"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detect if not specified)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch without pretrained weights"
    )

    args = parser.parse_args()

    train_detector(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=args.resume,
        pretrained=not args.no_pretrained
    )


if __name__ == "__main__":
    main()

