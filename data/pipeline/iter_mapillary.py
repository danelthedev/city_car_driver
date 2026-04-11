from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .taxonomy import MAPILLARY_PREFIX_TO_CLASS_ID, SPEED_LIMIT_TO_CLASS_ID
from .config import ClassManifest, PipelineConfig
from .yolo_utils import (
    counter_to_dict,
    ensure_dataset_layout,
    format_yolo_line,
    load_split_keys,
    prepare_dataset_dir,
    write_dataset_yaml,
    write_label_file,
    xyxy_to_yolo,
)


def normalize_mapillary_label(label: str) -> str:
    normalized = label.strip().lower()
    normalized = re.sub(r"--g\d+$", "", normalized)
    normalized = normalized.replace("_", "-")
    return normalized


def map_mapillary_label(label: str) -> Tuple[Optional[int], str]:
    normalized = normalize_mapillary_label(label)
    if not normalized:
        return None, "empty-label"

    speed_limit_match = re.search(r"maximum-speed-limit-(\d+)", normalized)
    if speed_limit_match:
        speed = int(speed_limit_match.group(1))
        class_id = SPEED_LIMIT_TO_CLASS_ID.get(speed)
        if class_id is not None:
            return class_id, "mapped"
        return None, "unmapped-speed-limit"

    for prefix, class_id in MAPILLARY_PREFIX_TO_CLASS_ID:
        if normalized.startswith(prefix):
            return class_id, "mapped"

    if normalized.startswith("regulatory--yield"):
        return 13, "mapped-fallback"
    if normalized.startswith("regulatory--stop"):
        return 14, "mapped-fallback"
    if normalized.startswith("regulatory--no-entry"):
        return 17, "mapped-fallback"
    if normalized.startswith("regulatory--keep-right"):
        return 38, "mapped-fallback"
    if normalized.startswith("regulatory--keep-left"):
        return 39, "mapped-fallback"
    if normalized.startswith("regulatory--roundabout"):
        return 40, "mapped-fallback"

    return None, "unmapped-label"


def is_usable_mapillary_object(obj: Dict[str, Any], cfg: PipelineConfig) -> Tuple[bool, str]:
    properties = obj.get("properties", {})
    if cfg.filter_ambiguous_mapillary and bool(properties.get("ambiguous", False)):
        return False, "ambiguous"
    if cfg.filter_dummy_mapillary and bool(properties.get("dummy", False)):
        return False, "dummy"
    if cfg.filter_out_of_frame_mapillary and bool(properties.get("out-of-frame", False)):
        return False, "out-of-frame"

    bbox = obj.get("bbox", {})
    required = {"xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(bbox):
        return False, "missing-bbox"

    try:
        xmin = float(bbox["xmin"])
        ymin = float(bbox["ymin"])
        xmax = float(bbox["xmax"])
        ymax = float(bbox["ymax"])
    except (TypeError, ValueError):
        return False, "non-numeric-bbox"

    if xmin >= xmax or ymin >= ymax:
        return False, "invalid-bbox"
    return True, "usable"


def find_mapillary_image(images_dir: Path, key: str) -> Optional[Path]:
    direct = images_dir / key
    if direct.exists() and direct.is_file():
        return direct

    for extension in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{key}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def prepare_iter_mapillary_backbone_dataset(
    cfg: PipelineConfig,
    manifest: ClassManifest,
) -> Dict[str, Any]:
    dataset_dir = cfg.output_root / "iter_mapillary_backbone"
    prepare_dataset_dir(dataset_dir, cfg.overwrite_output)
    ensure_dataset_layout(dataset_dir, ("train", "val", "test"))

    class_counts: Counter = Counter()
    dropped: Counter = Counter()
    processed_images = 0
    kept_objects = 0
    train_backgrounds: List[Tuple[Path, Path]] = []

    for split in ("train", "val", "test"):
        source_images_dir = cfg.mapillary_root / split / "images"
        split_keys = load_split_keys(cfg.mapillary_splits_dir / f"{split}.txt")

        for key in split_keys:
            source_image = find_mapillary_image(source_images_dir, key)
            source_annotation = cfg.mapillary_annotations_dir / f"{key}.json"
            if source_image is None or not source_annotation.exists():
                dropped["missing-mapillary-file"] += 1
                continue

            try:
                annotation = json.loads(source_annotation.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                dropped["invalid-mapillary-json"] += 1
                continue

            image_w = int(annotation.get("width", 0))
            image_h = int(annotation.get("height", 0))
            if image_w <= 0 or image_h <= 0:
                fallback = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
                if fallback is None:
                    dropped["unreadable-mapillary-image"] += 1
                    continue
                image_h, image_w = fallback.shape[:2]

            labels: List[str] = []
            for obj in annotation.get("objects", []):
                usable, reason = is_usable_mapillary_object(obj, cfg)
                if not usable:
                    dropped[reason] += 1
                    continue

                class_id, map_reason = map_mapillary_label(str(obj.get("label", "")))
                if class_id is None:
                    dropped[map_reason] += 1
                    continue

                bbox = obj.get("bbox", {})
                yolo_bbox = xyxy_to_yolo(
                    x1=float(bbox["xmin"]),
                    y1=float(bbox["ymin"]),
                    x2=float(bbox["xmax"]),
                    y2=float(bbox["ymax"]),
                    image_w=image_w,
                    image_h=image_h,
                    min_pixels=cfg.min_bbox_pixels,
                )
                if yolo_bbox is None:
                    dropped["bbox-too-small"] += 1
                    continue

                labels.append(format_yolo_line(class_id, yolo_bbox))
                class_counts[class_id] += 1
                kept_objects += 1

            destination_image = dataset_dir / "images" / split / source_image.name
            destination_label = dataset_dir / "labels" / split / f"{source_image.stem}.txt"
            shutil.copy2(source_image, destination_image)
            write_label_file(destination_label, labels, cfg.keep_empty_labels)

            if split == "train":
                train_backgrounds.append((destination_image, destination_label))

            processed_images += 1

    yaml_path = write_dataset_yaml(dataset_dir, manifest.names, include_test=True)
    print(
        "[iter_mapillary_backbone] images=",
        processed_images,
        "kept_objects=",
        kept_objects,
        "dropped_objects=",
        int(sum(dropped.values())),
    )

    return {
        "dataset_dir": str(dataset_dir),
        "yaml_path": str(yaml_path),
        "validate_splits": ["train", "val", "test"],
        "train_backgrounds": train_backgrounds,
        "report": {
            "processed_images": processed_images,
            "kept_objects": kept_objects,
            "class_counts": counter_to_dict(class_counts),
            "dropped_objects": counter_to_dict(dropped),
            "train_backgrounds": len(train_backgrounds),
            "copy_only": True,
        },
    }


def collect_train_backgrounds(mapillary_dataset_dir: Path) -> List[Tuple[Path, Path]]:
    images_dir = mapillary_dataset_dir / "images" / "train"
    labels_dir = mapillary_dataset_dir / "labels" / "train"
    backgrounds: List[Tuple[Path, Path]] = []
    if not images_dir.exists():
        return backgrounds

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        backgrounds.append((image_path, label_path))
    return backgrounds
