"""
Train ResNet-18 classifier on GTSRB dataset.
"""
from modules.traffic_signs.resnet_classifier import ResnetClassifier
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Create output directory
    Path('models/resnet_classifier').mkdir(parents=True, exist_ok=True)

    # Initialize classifier
    print("Initializing ResNet-18 classifier...")
    classifier = ResnetClassifier(num_classes=43, use_half=False)  # No half precision during training

    # Train the model
    print("\nStarting training...")
    history = classifier.train(
        train_data='data/datasets/GTSRB',
        epochs=30,
        batch_size=128,
        learning_rate=0.001
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = classifier.evaluate('data/datasets/GTSRB')

    # Save final model
    classifier.save_model('models/resnet_classifier/final_model.pth')

    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Final Test Accuracy: {metrics['accuracy']:.2f}%")
    print("="*50)


if __name__ == '__main__':
    main()

