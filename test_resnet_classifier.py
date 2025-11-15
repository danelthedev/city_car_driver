"""
Test ResNet-18 classifier on sample images.
"""
from modules.traffic_signs.resnet_classifier import ResnetClassifier
import cv2
import time
from pathlib import Path


def test_single_image(classifier, image_path):
    """Test classifier on a single image."""
    print(f"\nClassifying: {image_path}")

    result = classifier.classify_from_path(image_path)

    print(f"  Class ID: {result['class_id']}")
    print(f"  Class Name: {result['class_name']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Inference Time: {result['inference_time_ms']:.2f} ms")

    return result


def test_tmp_folder(classifier, tmp_dir='tmp'):
    """Test classifier on all images in tmp folder."""
    print(f"\n{'='*60}")
    print(f"Classifying all images in {tmp_dir} folder...")
    print('='*60)

    results = classifier.classify_tmp_folder(tmp_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("CLASSIFICATION SUMMARY")
    print('='*60)

    for result in results:
        print(f"\n{result['image_name']}:")
        print(f"  Class: {result['class_name']} (ID: {result['class_id']})")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Time: {result['inference_time_ms']:.2f} ms")

    return results


def benchmark_speed(classifier, num_runs=100):
    """Benchmark classifier inference speed."""
    print(f"\n{'='*60}")
    print(f"Running speed benchmark ({num_runs} iterations)...")
    print('='*60)

    # Create dummy image
    dummy_image = None
    if Path('test.png').exists():
        dummy_image = cv2.imread('test.png')
    elif Path('tmp').exists() and list(Path('tmp').glob('*.png')):
        dummy_image = cv2.imread(str(list(Path('tmp').glob('*.png'))[0]))

    if dummy_image is None:
        print("No test image found. Skipping benchmark.")
        return

    times = []
    for i in range(num_runs):
        start = time.time()
        result = classifier.classify(dummy_image)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms

        if i == 0:
            print(f"First inference: {times[0]:.2f} ms (includes warmup)")

    # Calculate statistics (excluding first run)
    times = times[1:]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nBenchmark Results (excluding first run):")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Total runs: {num_runs}")

    if avg_time <= 200:
        print(f"\n✓ PERFORMANCE TARGET MET: {avg_time:.2f} ms <= 200 ms")
    else:
        print(f"\n✗ Performance target not met: {avg_time:.2f} ms > 200 ms")


def main():
    # Load trained model
    model_path = 'models/resnet_classifier/best_model.pth'

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_resnet_classifier.py")
        return

    print("Loading ResNet-18 classifier...")
    classifier = ResnetClassifier(model_path=model_path, use_half=True)

    # Test Mode 1: Classify from image path
    if Path('test.png').exists():
        test_single_image(classifier, 'test.png')

    # Test Mode 2: Classify all images in tmp folder
    if Path('tmp').exists():
        test_tmp_folder(classifier)
    else:
        print("\nNo 'tmp' folder found. Skipping folder classification test.")

    # Benchmark speed
    benchmark_speed(classifier, num_runs=100)

    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)


if __name__ == '__main__':
    main()

