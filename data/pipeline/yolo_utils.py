from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------

def read_yaml(path: Path) -> Dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


# ---------------------------------------------------------------------------
# Dataset directory helpers
# ---------------------------------------------------------------------------

def prepare_dataset_dir(dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset directory already exists: {dataset_dir}. "
                "Use overwrite mode or remove it manually."
            )
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)


def ensure_dataset_layout(dataset_dir: Path, splits: Sequence[str]) -> None:
    for split in splits:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# YOLO label file helpers
# ---------------------------------------------------------------------------

def write_label_file(path: Path, lines: Sequence[str], keep_empty: bool) -> None:
    if not lines and not keep_empty:
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def read_yolo_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None


def parse_yolo_line_payload(line: str) -> Optional[Tuple[int, str]]:
    """Parse a YOLO detection line and return (class_id, bbox_payload_string)."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        source_id = int(parts[0])
        float(parts[1])
        float(parts[2])
        float(parts[3])
        float(parts[4])
    except ValueError:
        return None
    return source_id, " ".join(parts[1:])


def format_yolo_line(class_id: int, bbox: Tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = bbox
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ---------------------------------------------------------------------------
# Bounding-box geometry helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int,
    min_pixels: float,
) -> Optional[Tuple[float, float, float, float]]:
    x1 = clamp(x1, 0.0, float(image_w))
    y1 = clamp(y1, 0.0, float(image_h))
    x2 = clamp(x2, 0.0, float(image_w))
    y2 = clamp(y2, 0.0, float(image_h))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_pixels or box_h < min_pixels:
        return None

    x_center = ((x1 + x2) / 2.0) / float(image_w)
    y_center = ((y1 + y2) / 2.0) / float(image_h)
    width = box_w / float(image_w)
    height = box_h / float(image_h)
    return x_center, y_center, width, height


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_w: int,
    image_h: int,
) -> Tuple[float, float, float, float]:
    half_w = (width * float(image_w)) / 2.0
    half_h = (height * float(image_h)) / 2.0
    cx = x_center * float(image_w)
    cy = y_center * float(image_h)
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def box_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def lines_to_boxes(
    lines: Sequence[str],
    image_w: int,
    image_h: int,
) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for line in lines:
        parsed = parse_yolo_line(line)
        if parsed is None:
            continue
        _, x_center, y_center, width, height = parsed
        boxes.append(yolo_to_xyxy(x_center, y_center, width, height, image_w, image_h))
    return boxes


# ---------------------------------------------------------------------------
# Dataset YAML and split utilities
# ---------------------------------------------------------------------------

def write_dataset_yaml(
    dataset_dir: Path,
    class_names: Sequence[str],
    include_test: bool,
    train_relpath: str = "images/train",
    val_relpath: str = "images/val",
    test_relpath: str = "images/test",
) -> Path:
    data: Dict[str, object] = {
        "path": str(dataset_dir.resolve()),
        "train": train_relpath,
        "val": val_relpath,
        "nc": len(class_names),
        "names": list(class_names),
    }
    if include_test:
        data["test"] = test_relpath

    yaml_path = dataset_dir / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return yaml_path


def load_split_keys(split_file: Path) -> List[str]:
    if not split_file.exists():
        return []
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------

def validate_dataset(
    dataset_dir: Path,
    splits: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    split_reports: Dict[str, Any] = {}
    total_missing_labels = 0
    total_invalid_lines = 0

    for split in splits:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        image_files = (
            sorted(path for path in images_dir.iterdir() if path.is_file())
            if images_dir.exists()
            else []
        )
        missing_labels = 0
        invalid_lines = 0
        class_counts: Counter = Counter()

        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels += 1
                continue

            for line in read_yolo_lines(label_path):
                parsed = parse_yolo_line(line)
                if parsed is None:
                    invalid_lines += 1
                    continue

                class_id, _, _, width, height = parsed
                if class_id < 0 or class_id >= len(class_names):
                    invalid_lines += 1
                    continue
                if width <= 0.0 or height <= 0.0:
                    invalid_lines += 1
                    continue

                class_counts[class_id] += 1

        total_missing_labels += missing_labels
        total_invalid_lines += invalid_lines
        split_reports[split] = {
            "images": len(image_files),
            "missing_labels": missing_labels,
            "invalid_label_lines": invalid_lines,
            "class_counts": counter_to_dict(class_counts),
        }

    return {
        "dataset_dir": str(dataset_dir),
        "splits": split_reports,
        "total_missing_labels": total_missing_labels,
        "total_invalid_label_lines": total_invalid_lines,
    }


def existing_iterative_dataset_is_usable(
    dataset_dir: Path,
    required_splits: Sequence[str],
) -> bool:
    if not dataset_dir.exists() or not (dataset_dir / "dataset.yaml").exists():
        return False

    for split in required_splits:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            return False

    return True
