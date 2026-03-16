import os
import glob
import shutil
import random
import cv2
import yaml

# Standard 43 classes for German Traffic Sign Recognition Benchmark (GTSRB/GTSDB)
GTSDB_CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons', 
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left', 
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road', 
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 
    40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'gtsdb', 'FullIJCNN2013')
    output_dir = os.path.join(base_dir, 'gtsdb_yolo')
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    gt_file = os.path.join(source_dir, 'gt.txt')
    
    # Read annotations
    annotations = {}
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 6:
                    filename, xmin, ymin, xmax, ymax, cls_id = parts
                    if filename not in annotations:
                        annotations[filename] = []
                    annotations[filename].append((
                        float(xmin), float(ymin), float(xmax), float(ymax), int(cls_id)
                    ))

    # All images (GTSDB has 900 images from 00000.ppm to 00899.ppm)
    all_images = glob.glob(os.path.join(source_dir, '*.ppm'))
    random.seed(42)
    random.shuffle(all_images)
    
    # Split: 80% train, 20% val
    split_idx = int(0.8 * len(all_images))
    splits = {
        'train': all_images[:split_idx],
        'val': all_images[split_idx:]
    }
    
    print(f"Total images found: {len(all_images)}")
    
    for split_name, img_list in splits.items():
        print(f"Processing {split_name} split...")
        for img_path in img_list:
            base_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(base_name)[0]
            
            # Read image to get dimensions and convert to JPG (YOLO standard)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            
            # Save as JPG
            dest_img = os.path.join(output_dir, 'images', split_name, f"{name_no_ext}.jpg")
            cv2.imwrite(dest_img, img)
            
            # Prepare YOLO labels
            yolo_labels = []
            if base_name in annotations:
                for (xmin, ymin, xmax, ymax, cls_id) in annotations[base_name]:
                    # YOLO format: normalized x_center, y_center, width, height
                    x_center = ((xmin + xmax) / 2.0) / w
                    y_center = ((ymin + ymax) / 2.0) / h
                    width = (xmax - xmin) / w
                    height = (ymax - ymin) / h
                    
                    yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Write label file (even if empty, YOLO uses empty txt files for background images!)
            dest_label = os.path.join(output_dir, 'labels', split_name, f"{name_no_ext}.txt")
            with open(dest_label, 'w') as f:
                if yolo_labels:
                    f.write('\n'.join(yolo_labels))

    # Create dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    dataset_dict = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': GTSDB_CLASSES
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f, default_flow_style=False, sort_keys=False)
        
    print(f"GTSDB formatting complete! Dataset configuration saved to: {yaml_path}")

if __name__ == "__main__":
    main()
