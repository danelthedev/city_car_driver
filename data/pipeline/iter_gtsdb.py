from __future__ import annotations

import csv
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from .taxonomy import BASE_CLASS_NAMES
from .config import ClassManifest, PipelineConfig
from .yolo_utils import (
    box_iou,
    counter_to_dict,
    ensure_dataset_layout,
    format_yolo_line,
    lines_to_boxes,
    prepare_dataset_dir,
    read_yolo_lines,
    write_dataset_yaml,
    write_label_file,
    xyxy_to_yolo,
)


@dataclass(frozen=True)
class CropSample:
    image_path: Path
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


def parse_gtsdb_annotations(gt_path: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    grouped: Dict[str, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    with gt_path.open("r", encoding="utf-8") as handle:
        for row in handle:
            row = row.strip()
            if not row:
                continue

            parts = row.split(";")
            if len(parts) != 6:
                continue

            image_name = parts[0]
            try:
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                class_id = int(parts[5])
            except ValueError:
                continue

            if class_id < 0 or class_id >= len(BASE_CLASS_NAMES):
                continue
            grouped[image_name].append((class_id, x1, y1, x2, y2))
    return grouped


def parse_gtsrb_csv(csv_path: Path, gtsrb_root: Path) -> Dict[int, List[CropSample]]:
    grouped: Dict[int, List[CropSample]] = defaultdict(list)
    if not csv_path.exists():
        return grouped

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                class_id = int(row["ClassId"])
                width = int(row["Width"])
                height = int(row["Height"])
                x1 = int(row["Roi.X1"])
                y1 = int(row["Roi.Y1"])
                x2 = int(row["Roi.X2"]) + 1
                y2 = int(row["Roi.Y2"]) + 1
                relative_path = row["Path"].replace("\\", "/")
            except (KeyError, ValueError):
                continue

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))

            image_path = gtsrb_root / relative_path
            if not image_path.exists():
                continue
            if class_id < 0 or class_id >= len(BASE_CLASS_NAMES):
                continue

            grouped[class_id].append(
                CropSample(
                    image_path=image_path,
                    class_id=class_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return grouped


def build_gtsrb_crop_bank(cfg: PipelineConfig) -> Dict[int, List[CropSample]]:
    bank: Dict[int, List[CropSample]] = defaultdict(list)
    sources = [cfg.gtsrb_train_csv]
    if cfg.include_gtsrb_test:
        sources.append(cfg.gtsrb_test_csv)

    for source in sources:
        parsed = parse_gtsrb_csv(source, cfg.gtsrb_root)
        for class_id, samples in parsed.items():
            bank[class_id].extend(samples)

    print("[GTSRB] crop bank sizes:", {k: len(v) for k, v in sorted(bank.items()) if v})
    return bank


def prepare_crop_patch(
    sample: CropSample,
    background_w: int,
    background_h: int,
    cfg: PipelineConfig,
    rng: random.Random,
) -> Optional[Tuple[int, int, Any]]:
    crop_image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if crop_image is None:
        return None

    roi = crop_image[sample.y1 : sample.y2, sample.x1 : sample.x2]
    if roi.size == 0:
        return None

    target_max_side = int(
        min(background_w, background_h)
        * rng.uniform(cfg.synthetic_min_rel_size, cfg.synthetic_max_rel_size)
    )
    target_max_side = max(12, target_max_side)

    base_h, base_w = roi.shape[:2]
    scale = float(target_max_side) / float(max(base_w, base_h))
    patch_w = max(8, int(round(base_w * scale)))
    patch_h = max(8, int(round(base_h * scale)))
    if patch_w >= background_w or patch_h >= background_h:
        return None

    patch = cv2.resize(roi, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)

    angle = rng.uniform(-12.0, 12.0)
    center = (patch_w / 2.0, patch_h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    patch = cv2.warpAffine(
        patch,
        matrix,
        (patch_w, patch_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-15.0, 15.0)
    patch = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)
    return patch_w, patch_h, patch


def find_valid_placement(
    patch_w: int,
    patch_h: int,
    background_w: int,
    background_h: int,
    existing_boxes: Sequence[Tuple[float, float, float, float]],
    cfg: PipelineConfig,
    rng: random.Random,
) -> Optional[Tuple[int, int, int, int]]:
    if patch_w >= background_w or patch_h >= background_h:
        return None

    for _ in range(cfg.synthetic_max_attempts):
        x1 = rng.randint(0, background_w - patch_w)
        y1 = rng.randint(0, background_h - patch_h)
        x2 = x1 + patch_w
        y2 = y1 + patch_h
        candidate = (float(x1), float(y1), float(x2), float(y2))

        if all(box_iou(candidate, existing) <= cfg.synthetic_max_iou for existing in existing_boxes):
            return x1, y1, x2, y2
    return None


def generate_synthetic_samples(
    cfg: PipelineConfig,
    rng: random.Random,
    dataset_dir: Path,
    train_backgrounds: Sequence[Tuple[Path, Path]],
    train_counts: Counter,
    crop_bank: Dict[int, List[CropSample]],
) -> Dict[str, Any]:
    nonzero_counts = [count for count in train_counts.values() if count > 0]
    median_count = int(statistics.median(nonzero_counts)) if nonzero_counts else cfg.synthetic_min_instances
    target_count = max(cfg.synthetic_min_instances, median_count)

    planned_per_class: Dict[int, int] = {}
    for class_id in range(len(BASE_CLASS_NAMES)):
        if not crop_bank.get(class_id):
            continue
        deficit = target_count - int(train_counts.get(class_id, 0))
        if deficit > 0:
            planned_per_class[class_id] = min(deficit, cfg.synthetic_max_per_class)

    generated_per_class: Counter = Counter()
    failures: Counter = Counter()
    image_index = 0

    if not train_backgrounds:
        return {
            "target_count": target_count,
            "planned_per_class": {},
            "generated_per_class": {},
            "failures": {"no-mapillary-backgrounds": 1},
            "generated_images": 0,
        }

    for class_id, planned_count in sorted(planned_per_class.items()):
        class_crops = crop_bank.get(class_id, [])
        if not class_crops:
            continue

        for _ in range(planned_count):
            background_image_path, background_label_path = rng.choice(train_backgrounds)
            background = cv2.imread(str(background_image_path), cv2.IMREAD_COLOR)
            if background is None:
                failures["unreadable-background"] += 1
                continue

            background_h, background_w = background.shape[:2]
            background_labels = read_yolo_lines(background_label_path)
            existing_boxes = lines_to_boxes(background_labels, background_w, background_h)

            crop_sample = rng.choice(class_crops)
            patch_data = prepare_crop_patch(crop_sample, background_w, background_h, cfg, rng)
            if patch_data is None:
                failures["invalid-crop"] += 1
                continue

            patch_w, patch_h, patch = patch_data
            placement = find_valid_placement(
                patch_w=patch_w,
                patch_h=patch_h,
                background_w=background_w,
                background_h=background_h,
                existing_boxes=existing_boxes,
                cfg=cfg,
                rng=rng,
            )
            if placement is None:
                failures["no-valid-placement"] += 1
                continue

            x1, y1, x2, y2 = placement
            background[y1:y2, x1:x2] = patch

            new_bbox = xyxy_to_yolo(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                image_w=background_w,
                image_h=background_h,
                min_pixels=1.0,
            )
            if new_bbox is None:
                failures["invalid-generated-bbox"] += 1
                continue

            output_stem = f"syn_{class_id:02d}_{image_index:06d}"
            output_image = dataset_dir / "images" / "train" / f"{output_stem}.jpg"
            output_label = dataset_dir / "labels" / "train" / f"{output_stem}.txt"

            all_labels = list(background_labels)
            all_labels.append(format_yolo_line(class_id, new_bbox))

            cv2.imwrite(str(output_image), background, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            write_label_file(output_label, all_labels, keep_empty=True)

            generated_per_class[class_id] += 1
            image_index += 1

    return {
        "target_count": int(target_count),
        "planned_per_class": {str(k): int(v) for k, v in sorted(planned_per_class.items())},
        "generated_per_class": counter_to_dict(generated_per_class),
        "failures": counter_to_dict(failures),
        "generated_images": int(image_index),
    }


def prepare_iter_gtsdb_plus_synthetic_dataset(
    cfg: PipelineConfig,
    manifest: ClassManifest,
    rng: random.Random,
    train_backgrounds: Sequence[Tuple[Path, Path]],
) -> Dict[str, Any]:
    dataset_dir = cfg.output_root / "iter_gtsdb_plus_synthetic"
    prepare_dataset_dir(dataset_dir, cfg.overwrite_output)
    ensure_dataset_layout(dataset_dir, ("train", "val"))

    annotations = parse_gtsdb_annotations(cfg.gtsdb_gt_path)
    image_paths = sorted(cfg.gtsdb_root.glob("*.ppm"))
    if not image_paths:
        raise FileNotFoundError(f"No GTSDB .ppm images found in {cfg.gtsdb_root}")

    shuffled = list(image_paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * cfg.gtsdb_val_ratio))
    val_set = {path.name for path in shuffled[:val_count]}

    class_counts_train: Counter = Counter()
    class_counts_val: Counter = Counter()
    processed_images = 0

    for image_path in image_paths:
        split = "val" if image_path.name in val_set else "train"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        image_h, image_w = image.shape[:2]
        stem = f"gtsdb_{image_path.stem}"
        destination_image = dataset_dir / "images" / split / f"{stem}.jpg"
        destination_label = dataset_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(destination_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        labels: List[str] = []
        for class_id, x1, y1, x2, y2 in annotations.get(image_path.name, []):
            yolo_bbox = xyxy_to_yolo(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_w=image_w,
                image_h=image_h,
                min_pixels=cfg.min_bbox_pixels,
            )
            if yolo_bbox is None:
                continue

            labels.append(format_yolo_line(class_id, yolo_bbox))
            if split == "train":
                class_counts_train[class_id] += 1
            else:
                class_counts_val[class_id] += 1

        write_label_file(destination_label, labels, cfg.keep_empty_labels)
        processed_images += 1

    crop_bank = build_gtsrb_crop_bank(cfg)
    synthetic_stats = generate_synthetic_samples(
        cfg=cfg,
        rng=rng,
        dataset_dir=dataset_dir,
        train_backgrounds=train_backgrounds,
        train_counts=class_counts_train,
        crop_bank=crop_bank,
    )

    yaml_path = write_dataset_yaml(dataset_dir, manifest.names, include_test=False)

    print(
        "[iter_gtsdb_plus_synthetic] images=",
        processed_images,
        "train_objects=",
        int(sum(class_counts_train.values())),
        "val_objects=",
        int(sum(class_counts_val.values())),
        "synthetic_images=",
        int(synthetic_stats["generated_images"]),
    )

    return {
        "dataset_dir": str(dataset_dir),
        "yaml_path": str(yaml_path),
        "validate_splits": ["train", "val"],
        "report": {
            "processed_images": processed_images,
            "train_class_counts": counter_to_dict(class_counts_train),
            "val_class_counts": counter_to_dict(class_counts_val),
            "synthetic": synthetic_stats,
        },
    }
