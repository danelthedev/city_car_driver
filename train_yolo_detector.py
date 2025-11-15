"""
Training script for YOLO Traffic Sign Detector on GTSDB dataset.
This script demonstrates how to train the model using GPU (CUDA).
"""

from modules.traffic_signs.yolo_detector import YoloTrafficSignDetector
from pathlib import Path

def main():
    """Main training pipeline for YOLO traffic sign detector."""

    # Initialize the detector
    print("=" * 60)
    print("YOLO Traffic Sign Detector - Training Pipeline")
    print("=" * 60)

    detector = YoloTrafficSignDetector(confidence_threshold=0.5)

    # Path to the GTSDB_YOLO dataset YAML file
    data_yaml = "data/datasets/GTSDB_YOLO/data.yaml"

    # Check if dataset exists
    if not Path(data_yaml).exists():
        print(f"Error: Dataset YAML file not found at {data_yaml}")
        return

    print(f"\nDataset configuration: {data_yaml}")

    # Training parameters
    training_params = {
        'epochs': 100,           # Number of training epochs
        'batch_size': 16,        # Batch size (adjust based on GPU memory)
        'imgsz': 640,            # Image size
        'patience': 20,          # Early stopping patience
        'lr0': 0.01,             # Initial learning rate
        'lrf': 0.01,             # Final learning rate
        'project': 'models',     # Save directory
        'name': 'gtsdb_detector', # Experiment name
        'exist_ok': True,        # Overwrite existing
        'plots': True,           # Generate plots
        'save': True,            # Save checkpoints
        'val': True,             # Validate during training
    }

    print("\nTraining Parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")

    # Train the model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")

    history = detector.train(
        train_data=data_yaml,
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        **{k: v for k, v in training_params.items() if k not in ['epochs', 'batch_size']}
    )

    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)

    # Save the best model to models folder
    best_model_path = history.get('best_model_path')
    if best_model_path:
        print(f"\nBest model saved at: {best_model_path}")

        # Copy best model to a permanent location
        final_model_path = Path("models") / "gtsdb_best.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Load best model and save it
        detector.load_model(best_model_path)
        detector.save_model(str(final_model_path))

        print(f"Model also saved to: {final_model_path}")

    # Evaluate the model
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60 + "\n")

    metrics = detector.evaluate(data_yaml)

    print("\nFinal Evaluation Metrics:")
    print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5-0.95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

    # Get class names
    class_names = detector.get_class_names()
    print(f"\nModel can detect {len(class_names)} traffic sign classes")

    print("\n" + "=" * 60)
    print("Training Pipeline Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

