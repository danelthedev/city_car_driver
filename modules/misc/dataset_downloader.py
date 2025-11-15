import os
import zipfile
import shutil
import csv
from pathlib import Path
import requests
from tqdm import tqdm
import cv2
import numpy as np


def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
            desc=destination,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_gtsrb(data_dir="data/datasets"):
    """Download GTSRB dataset"""
    print("Downloading GTSRB dataset...")

    gtsrb_dir = f"{data_dir}/GTSRB"
    Path(gtsrb_dir).mkdir(parents=True, exist_ok=True)

    # GTSRB URLs
    train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    test_labels_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

    # Download files
    train_zip = gtsrb_dir + "/train.zip"
    test_zip = gtsrb_dir + "/test.zip"
    test_labels_zip = gtsrb_dir + "/test_labels.zip"

    if not Path(train_zip).exists():
        download_file(train_url, train_zip)

    if not Path(test_zip).exists():
        download_file(test_url, test_zip)

    if not Path(test_labels_zip).exists():
        download_file(test_labels_url, test_labels_zip)

    # Extract files
    print("Extracting GTSRB files...")
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall(gtsrb_dir)
    os.remove(train_zip)

    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        zip_ref.extractall(gtsrb_dir)
    os.remove(test_zip)

    with zipfile.ZipFile(test_labels_zip, 'r') as zip_ref:
        zip_ref.extractall(gtsrb_dir)
    os.remove(test_labels_zip)

    print("GTSRB download complete!")
    return gtsrb_dir


def download_gtsdb(data_dir="data/datasets"):
    """Download GTSDB dataset"""
    print("Downloading GTSDB dataset...")

    gtsdb_dir = data_dir + "/GTSDB"
    Path(gtsdb_dir).mkdir(parents=True, exist_ok=True)

    # GTSDB URLs
    train_url = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"

    # Download file
    train_zip = gtsdb_dir + "/FullIJCNN2013.zip"

    if not Path(train_zip).exists():
        download_file(train_url, train_zip)

    # Extract files
    print("Extracting GTSDB files...")
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall(gtsdb_dir)
    os.remove(train_zip)

    print("GTSDB download complete!")
    return gtsdb_dir


def convert_gtsdb_to_yolo(gtsdb_dir, output_dir="data/yolo_dataset"):
    """Convert GTSDB to YOLO format"""
    print("Converting GTSDB to YOLO format...")

    gtsdb_path = Path(gtsdb_dir)
    output_path = Path(output_dir)

    # Create output directories
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Read annotations
    gt_file = gtsdb_path / "FullIJCNN2013" / "gt.txt"

    annotations = {}
    with open(gt_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 6:
                continue
            filename = row[0]
            x1, y1, x2, y2 = map(int, row[1:5])
            class_id = int(row[5])

            if filename not in annotations:
                annotations[filename] = []
            annotations[filename].append((x1, y1, x2, y2, class_id))

    # Split dataset (80% train, 20% val)
    image_files = list(annotations.keys())
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def process_split(files, split_name):
        for filename in tqdm(files, desc=f"Processing {split_name}"):
            img_path = gtsdb_path / "FullIJCNN2013" / filename

            if not img_path.exists():
                continue

            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Copy image
            dst_img = output_path / "images" / split_name / filename
            shutil.copy(img_path, dst_img)

            # Create YOLO label file
            label_file = output_path / "labels" / split_name / f"{Path(filename).stem}.txt"

            with open(label_file, 'w') as f:
                for x1, y1, x2, y2, class_id in annotations[filename]:
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    process_split(train_files, "train")
    process_split(val_files, "val")

    # Create data.yaml
    yaml_content = f"""
path: {output_path.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 43

# Class names (German Traffic Sign classes)
names:
  0: speed_limit_20
  1: speed_limit_30
  2: speed_limit_50
  3: speed_limit_60
  4: speed_limit_70
  5: speed_limit_80
  6: end_speed_limit_80
  7: speed_limit_100
  8: speed_limit_120
  9: no_overtaking
  10: no_overtaking_trucks
  11: priority_at_next_intersection
  12: priority_road
  13: give_way
  14: stop
  15: no_traffic_both_ways
  16: no_trucks
  17: no_entry
  18: danger
  19: bend_left
  20: bend_right
  21: bend
  22: uneven_road
  23: slippery_road
  24: road_narrows
  25: construction
  26: traffic_signal
  27: pedestrian_crossing
  28: school_crossing
  29: cycles_crossing
  30: snow
  31: animals
  32: end_restrictions
  33: go_right
  34: go_left
  35: go_straight
  36: go_right_or_straight
  37: go_left_or_straight
  38: keep_right
  39: keep_left
  40: roundabout
  41: end_no_overtaking
  42: end_no_overtaking_trucks
"""

    with open(output_path / "data.yaml", 'w') as f:
        f.write(yaml_content.strip())

    print(f"YOLO dataset created at {output_path}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")


# Create data directory
data_dir = "E:/Dev/Dizertatie/data/datasets"

try:
    gtsrb_dir = download_gtsrb(data_dir)
except Exception as e:
    print(f"Error downloading GTSRB: {e}")

try:
    gtsdb_dir = download_gtsdb(data_dir)
except Exception as e:
    print(f"Error downloading GTSDB: {e}")

gtsdb_dir = "E:/Dev/Dizertatie/data/datasets"
try:
    convert_gtsdb_to_yolo(gtsdb_dir + "GTSDB", gtsdb_dir + "/GTSDB_YOLO")
    print("Dataset conversion complete!")
except Exception as e:
    print(f"Error converting dataset: {e}")
    import traceback
    traceback.print_exc()

