"""
Script to convert GTSDB_YOLO dataset to a single-class dataset.
All traffic signs will be labeled as class 0 regardless of their type.
"""

import shutil
from pathlib import Path


def convert_label_to_single_class(label_path, output_path):
    """
    Convert a YOLO label file to single class (class 0).

    Args:
        label_path: Path to the original label file
        output_path: Path to save the converted label file
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # YOLO format: class x_center y_center width height
            parts = line.split()
            if len(parts) >= 5:
                # Change class to 0, keep the rest
                parts[0] = '0'
                converted_lines.append(' '.join(parts) + '\n')

    with open(output_path, 'w') as f:
        f.writelines(converted_lines)


def convert_dataset(source_dir, target_dir):
    """
    Convert the entire GTSDB_YOLO dataset to a single-class dataset.

    Args:
        source_dir: Path to the source GTSDB_YOLO directory
        target_dir: Path to the target GTSDB_YOLO_1_CLASS directory
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory structure
    print(f"Creating directory structure at {target_path}")
    target_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for split in ['train', 'val']:
        (target_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process train and val splits
    for split in ['train', 'val']:
        source_images = source_path / 'images' / split
        source_labels = source_path / 'labels' / split
        target_images = target_path / 'images' / split
        target_labels = target_path / 'labels' / split

        if not source_images.exists():
            print(f"Warning: {source_images} does not exist, skipping...")
            continue

        print(f"\nProcessing {split} split...")

        # Get all image files
        image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))

        # Copy images and convert labels
        for img_file in image_files:
            # Copy image
            shutil.copy2(img_file, target_images / img_file.name)

            # Convert label
            label_file = source_labels / f"{img_file.stem}.txt"
            target_label_file = target_labels / f"{img_file.stem}.txt"

            if label_file.exists():
                convert_label_to_single_class(label_file, target_label_file)
            else:
                # Create empty label file if no labels exist
                target_label_file.touch()

        print(f"Processed {len(image_files)} images for {split} split")

    # Create data.yaml file
    create_data_yaml(target_path)

    print(f"\nConversion complete! Dataset saved to {target_path}")


def create_data_yaml(target_path):
    """
    Create a data.yaml file for the single-class dataset.

    Args:
        target_path: Path to the target GTSDB_YOLO_1_CLASS directory
    """
    yaml_content = f"""path: {str(target_path).replace(chr(92), '/')}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names
names:
  0: traffic_sign
"""

    yaml_path = target_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created data.yaml at {yaml_path}")


def main():
    """Main function to run the conversion."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    source_dir = project_root / 'data' / 'datasets' / 'GTSDB_YOLO'
    target_dir = project_root / 'data' / 'datasets' / 'GTSDB_YOLO_1_CLASS'

    print("=" * 60)
    print("GTSDB YOLO to Single-Class Converter")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("=" * 60)

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist!")
        return

    if target_dir.exists():
        response = input(f"\nTarget directory {target_dir} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
        print("Removing existing target directory...")
        shutil.rmtree(target_dir)

    # Run conversion
    convert_dataset(source_dir, target_dir)

    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

