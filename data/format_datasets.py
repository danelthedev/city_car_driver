from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import yaml


# Canonical 43-class base taxonomy used by the current checkpoints.
BASE_CLASS_NAMES: List[str] = [
    "speed-limit-20",
    "speed-limit-30",
    "speed-limit-50",
    "speed-limit-60",
    "speed-limit-70",
    "speed-limit-80",
    "restriction-ends-80",
    "speed-limit-100",
    "speed-limit-120",
    "no-overtaking",
    "no-overtaking-trucks",
    "priority-at-next-intersection",
    "priority-road",
    "give-way",
    "stop",
    "no-traffic-both-ways",
    "no-trucks",
    "no-entry",
    "danger",
    "bend-left",
    "bend-right",
    "bend",
    "uneven-road",
    "slippery-road",
    "road-narrows",
    "construction",
    "traffic-signal",
    "pedestrian-crossing",
    "school-crossing",
    "cycles-crossing",
    "snow",
    "animals",
    "restriction-ends",
    "go-right",
    "go-left",
    "go-straight",
    "go-right-or-straight",
    "go-left-or-straight",
    "keep-right",
    "keep-left",
    "roundabout",
    "restriction-ends-overtaking",
    "restriction-ends-overtaking-trucks",
]


# BSTLD label → unified traffic-light class name.
# Arrow variants are collapsed into their base color; ambiguous/unknown
# labels are intentionally omitted so they get dropped during conversion.
# Standalone 4-class taxonomy used exclusively by iter_bstld.
# Deliberately kept separate from the sign manifest so the two models
# (sign detector and traffic-light detector) remain independent.
BSTLD_CLASS_NAMES: List[str] = ["red", "yellow", "green", "off"]
BSTLD_CLASS_ID: Dict[str, int] = {name: idx for idx, name in enumerate(BSTLD_CLASS_NAMES)}

# Maps every Supervisely classTitle in the BSTLD export to one of the 4 classes.
# Arrow variants are collapsed into their base colour; unknown labels are
# omitted so they get dropped during conversion.
BSTLD_LABEL_MAP: Dict[str, str] = {
    # Base colours (as they appear in this Supervisely export)
    "red":               "red",
    "yellow":            "yellow",
    "green":             "green",
    "off":               "off",
    # Capitalised variants (original Bosch format, kept for safety)
    "Red":               "red",
    "RedLeft":           "red",
    "RedRight":          "red",
    "RedStraight":       "red",
    "RedStraightLeft":   "red",
    "Yellow":            "yellow",
    "Green":             "green",
    "GreenLeft":         "green",
    "GreenRight":        "green",
    "GreenStraight":     "green",
    "GreenStraightLeft": "green",
    "off":               "off",
}

# Conservative Romanian class mappings to existing base classes.
ROMANIAN_TO_BASE_CLASS: Dict[str, str] = {
    "forb_overtake": "no-overtaking",
    "forb_speed_over_20": "speed-limit-20",
    "forb_speed_over_30": "speed-limit-30",
    "forb_speed_over_50": "speed-limit-50",
    "forb_speed_over_60": "speed-limit-60",
    "forb_speed_over_70": "speed-limit-70",
    "forb_speed_over_80": "speed-limit-80",
    "forb_speed_over_100": "speed-limit-100",
    "forb_trucks": "no-trucks",
    "info_crosswalk": "pedestrian-crossing",
    "mand_left": "go-left",
    "mand_pass_left": "keep-left",
    "mand_pass_right": "keep-right",
    "mand_right": "go-right",
    "mand_roundabout": "roundabout",
    "mand_straigh_left": "go-left-or-straight",
    "mand_straight": "go-straight",
    "mand_straight_right": "go-right-or-straight",
    "prio_give_way": "give-way",
    "prio_priority_road": "priority-road",
    "prio_stop": "stop",
    "warn_children": "school-crossing",
    "warn_construction": "construction",
    "warn_crosswalk": "pedestrian-crossing",
    "warn_cyclists": "cycles-crossing",
    "warn_domestic_animals": "animals",
    "warn_other_dangers": "danger",
    "warn_poor_road_surface": "uneven-road",
    "warn_slippery_road": "slippery-road",
    "warn_traffic_light": "traffic-signal",
    "warn_wild_animals": "animals",
}


ROMANIAN_DROP_CLASS_NAMES: Set[str] = {
    "forb_speed_over_5",
    "forb_speed_over_10",
    "forb_speed_over_40",
    "forb_speed_over_90",
    "forb_speed_over_130",
}


# These Romanian classes must stay as explicit appended classes when present.
ROMANIAN_FORCE_APPEND_CLASS_NAMES: Set[str] = {
    "forb_ahead",
    "forb_left",
    "forb_right",
    "forb_stopping",
    "forb_u_turn",
    "forb_weight_over_3.5t",
    "forb_weight_over_7.5t",
    "info_bus_station",
    "info_highway",
    "info_one_way_traffic",
    "info_parking",
    "info_taxi_parking",
    "mand_bike_lane",
    "mand_left_right",
    "mand_pass_left_right",
    "warn_roundabout",
    "warn_speed_bumper",
    "warn_tram",
    "warn_two_way_traffic",
}


SPEED_LIMIT_TO_CLASS_ID = {
    20: 0,
    30: 1,
    50: 2,
    60: 3,
    70: 4,
    80: 5,
    100: 7,
    120: 8,
}


MAPILLARY_PREFIX_TO_CLASS_ID: List[Tuple[str, int]] = [
    ("regulatory--end-of-no-overtaking-by-trucks", 42),
    ("regulatory--end-of-no-overtaking", 41),
    ("regulatory--roundabout", 40),
    ("regulatory--keep-left", 39),
    ("regulatory--keep-right", 38),
    ("complementary--keep-left", 39),
    ("complementary--keep-right", 38),
    ("regulatory--go-straight-or-left", 37),
    ("regulatory--go-straight-or-right", 36),
    ("regulatory--go-straight", 35),
    ("regulatory--turn-left", 34),
    ("regulatory--turn-right", 33),
    ("complementary--go-left-or-straight", 37),
    ("complementary--go-right-or-straight", 36),
    ("complementary--go-straight", 35),
    ("complementary--go-left", 34),
    ("complementary--go-right", 33),
    ("regulatory--end-of-all-restrictions", 32),
    ("warning--animals", 31),
    ("warning--snow", 30),
    ("warning--bicycles-crossing", 29),
    ("warning--children", 28),
    ("warning--pedestrians-crossing", 27),
    ("warning--traffic-signals", 26),
    ("warning--construction", 25),
    ("warning--road-narrows", 24),
    ("warning--slippery-road", 23),
    ("warning--uneven-road", 22),
    ("warning--double-curve", 21),
    ("warning--curve-right", 20),
    ("warning--curve-left", 19),
    ("warning--danger", 18),
    ("regulatory--no-entry", 17),
    ("regulatory--no-trucks", 16),
    ("regulatory--no-vehicles", 15),
    ("regulatory--stop", 14),
    ("regulatory--yield", 13),
    ("regulatory--priority-road", 12),
    ("warning--priority-at-next-intersection", 11),
    ("regulatory--no-overtaking--trucks", 10),
    ("regulatory--no-overtaking", 9),
    ("regulatory--end-of-maximum-speed-limit-80", 6),
]


DATASET_TARGETS: Tuple[str, ...] = (
    "iter_mapillary_backbone",
    "iter_gtsdb_plus_synthetic",
    "iter_romanian",
    "iter_bstld",
)


DATASET_TARGET_ALIASES: Dict[str, str] = {
    "mapillary": "iter_mapillary_backbone",
    "mapillary_backbone": "iter_mapillary_backbone",
    "iter_mapillary_backbone": "iter_mapillary_backbone",
    "gtsdb": "iter_gtsdb_plus_synthetic",
    "gtsdb_synthetic": "iter_gtsdb_plus_synthetic",
    "gtsdb_plus_synthetic": "iter_gtsdb_plus_synthetic",
    "iter_gtsdb_plus_synthetic": "iter_gtsdb_plus_synthetic",
    "romanian": "iter_romanian",
    "romanian_finetune": "iter_romanian",
    "iter_romanian": "iter_romanian",
    "bstld": "iter_bstld",
    "iter_bstld": "iter_bstld",
}


TARGET_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "iter_gtsdb_plus_synthetic": ("iter_mapillary_backbone",),
}


@dataclass(frozen=True)
class CropSample:
    image_path: Path
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class ClassManifest:
    names: List[str]
    id_by_name: Dict[str, int]
    romanian_source_names: List[str]
    romanian_extra_names: List[str]


@dataclass
class PipelineConfig:
    data_root: Path
    output_root: Path
    seed: int = 42
    gtsdb_val_ratio: float = 0.20
    bstld_val_ratio: float = 0.15
    bstld_negatives: int = 2000
    overwrite_output: bool = True
    keep_empty_labels: bool = True
    include_gtsrb_test: bool = True
    filter_occluded_bstld: bool = True
    min_bbox_pixels: float = 2.0
    synthetic_min_instances: int = 250
    synthetic_max_per_class: int = 400
    synthetic_min_rel_size: float = 0.015
    synthetic_max_rel_size: float = 0.060
    synthetic_max_iou: float = 0.25
    synthetic_max_attempts: int = 35
    filter_ambiguous_mapillary: bool = True
    filter_dummy_mapillary: bool = True
    filter_out_of_frame_mapillary: bool = True

    @property
    def mapillary_root(self) -> Path:
        return self.data_root / "mapillary"

    @property
    def mapillary_annotations_dir(self) -> Path:
        return self.mapillary_root / "annotations"

    @property
    def mapillary_splits_dir(self) -> Path:
        return self.mapillary_root / "splits"

    @property
    def gtsdb_root(self) -> Path:
        return self.data_root / "gtsdb" / "FullIJCNN2013"

    @property
    def gtsdb_gt_path(self) -> Path:
        return self.gtsdb_root / "gt.txt"

    @property
    def gtsrb_root(self) -> Path:
        return self.data_root / "gtsrb"

    @property
    def gtsrb_train_csv(self) -> Path:
        return self.gtsrb_root / "Train.csv"

    @property
    def gtsrb_test_csv(self) -> Path:
        return self.gtsrb_root / "Test.csv"

    @property
    def romanian_root(self) -> Path:
        return self.data_root / "romanian_traffic_signs"

    @property
    def bstld_root(self) -> Path:
        return self.data_root / "bstld"

    @property
    def bstld_train_img_dir(self) -> Path:
        return self.bstld_root / "train" / "img"

    @property
    def bstld_train_ann_dir(self) -> Path:
        return self.bstld_root / "train" / "ann"

    @property
    def bstld_test_img_dir(self) -> Path:
        return self.bstld_root / "test" / "img"

    @property
    def bstld_test_ann_dir(self) -> Path:
        return self.bstld_root / "test" / "ann"


def parse_requested_datasets(raw_value: str) -> List[str]:
    tokens = [token.strip().lower() for token in re.split(r"[\s,]+", raw_value) if token.strip()]
    if not tokens or "all" in tokens:
        return list(DATASET_TARGETS)

    canonical = [DATASET_TARGET_ALIASES.get(token, token) for token in tokens]
    invalid = sorted({token for token in canonical if token not in DATASET_TARGETS})
    if invalid:
        allowed = ", ".join(DATASET_TARGETS)
        raise ValueError(f"Unknown dataset target(s): {', '.join(invalid)}. Allowed values: all, {allowed}")

    selected_set = set(canonical)
    return [target for target in DATASET_TARGETS if target in selected_set]


def expand_target_dependencies(requested: Sequence[str]) -> Tuple[List[str], List[str]]:
    selected: Set[str] = set(requested)
    changed = True
    while changed:
        changed = False
        for target in list(selected):
            for dependency in TARGET_DEPENDENCIES.get(target, ()):
                if dependency not in selected:
                    selected.add(dependency)
                    changed = True

    execution = [target for target in DATASET_TARGETS if target in selected]
    auto_added = [target for target in execution if target not in requested]
    return execution, auto_added


def read_yaml(path: Path) -> Dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def prepare_output_root(cfg: PipelineConfig) -> None:
    output_root = cfg.output_root.resolve()
    data_root = cfg.data_root.resolve()
    if output_root == data_root:
        raise ValueError("output_root must not be the same as data_root")
    output_root.mkdir(parents=True, exist_ok=True)


def prepare_dataset_dir(dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset directory already exists: {dataset_dir}. Use overwrite mode or remove it manually."
            )
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)


def ensure_dataset_layout(dataset_dir: Path, splits: Sequence[str]) -> None:
    for split in splits:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


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


def box_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
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


def lines_to_boxes(lines: Sequence[str], image_w: int, image_h: int) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for line in lines:
        parsed = parse_yolo_line(line)
        if parsed is None:
            continue
        _, x_center, y_center, width, height = parsed
        boxes.append(yolo_to_xyxy(x_center, y_center, width, height, image_w, image_h))
    return boxes


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


def parse_romanian_names_field(raw_names: Any) -> List[str]:
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]

    if isinstance(raw_names, dict):
        parsed: List[Tuple[int, str]] = []
        for raw_key, raw_value in raw_names.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            parsed.append((key, str(raw_value)))
        return [name for _, name in sorted(parsed, key=lambda item: item[0])]

    return []


def load_romanian_source_names(cfg: PipelineConfig) -> List[str]:
    config_path = cfg.romanian_root / "data.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Romanian source config: {config_path}")

    config = read_yaml(config_path)
    names = parse_romanian_names_field(config.get("names", []))
    declared_nc = int(config.get("nc", len(names)))
    if declared_nc != len(names):
        raise ValueError(f"Romanian data.yaml nc ({declared_nc}) does not match names length ({len(names)}).")
    return names


def build_class_manifest(cfg: PipelineConfig) -> ClassManifest:
    romanian_source_names = load_romanian_source_names(cfg)
    base_set = set(BASE_CLASS_NAMES)
    extra_names: List[str] = []
    seen_extras: Set[str] = set()
    for source_name in romanian_source_names:
        if source_name in ROMANIAN_FORCE_APPEND_CLASS_NAMES:
            if source_name not in seen_extras:
                seen_extras.add(source_name)
                extra_names.append(source_name)
            continue
        if source_name in ROMANIAN_DROP_CLASS_NAMES:
            continue
        if source_name in ROMANIAN_TO_BASE_CLASS:
            continue
        if source_name in base_set:
            continue
        if source_name not in seen_extras:
            seen_extras.add(source_name)
            extra_names.append(source_name)
    names = [*BASE_CLASS_NAMES, *extra_names]
    id_by_name = {name: idx for idx, name in enumerate(names)}
    return ClassManifest(
        names=names,
        id_by_name=id_by_name,
        romanian_source_names=romanian_source_names,
        romanian_extra_names=extra_names,
    )


def class_manifest_to_dict(manifest: ClassManifest) -> Dict[str, Any]:
    return {
        "nc": len(manifest.names),
        "names": manifest.names,
        "base_nc": len(BASE_CLASS_NAMES),
        "romanian_source_nc": len(manifest.romanian_source_names),
        "romanian_extra_nc": len(manifest.romanian_extra_names),
        "romanian_extra_names": manifest.romanian_extra_names,
        "romanian_force_append_classes": sorted(ROMANIAN_FORCE_APPEND_CLASS_NAMES),
    }


# ---------------------------------------------------------------------------
# BSTLD dataset support  (Supervisely export format)
# ---------------------------------------------------------------------------
# Expected layout:
#
#   bstld/
#     train/
#       img/          ← frame images (any extension)
#       ann/          ← one JSON per image, same stem
#     test/
#       img/
#       ann/
#     meta.json       ← class list (not used; we rely on BSTLD_LABEL_MAP)
#
# Supervisely annotation JSON schema (per image):
#   {
#     "size": {"height": H, "width": W},
#     "objects": [
#       {
#         "classTitle": "Green",
#         "points": {
#           "exterior": [[x1, y1], [x2, y2]],
#           "interior": []
#         }
#       }, ...
#     ]
#   }
# ---------------------------------------------------------------------------

@dataclass
class BstldFrame:
    """One annotated frame from the Supervisely BSTLD export."""
    image_path: Path        # absolute path to the image file
    ann_path: Path          # absolute path to the companion JSON annotation


def _collect_bstld_frames(img_dir: Path, ann_dir: Path) -> Tuple[List[BstldFrame], Counter]:
    """Pair every image in *img_dir* with its annotation JSON in *ann_dir*.

    Images without a matching annotation are counted as skipped; the reverse
    (annotation without an image) is silently ignored.
    """
    skipped: Counter = Counter()
    frames: List[BstldFrame] = []

    if not img_dir.exists():
        raise FileNotFoundError(f"BSTLD image directory not found: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"BSTLD annotation directory not found: {ann_dir}")

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file():
            continue
        # Supervisely names the annotation file as "<image_filename>.json",
        # e.g. "frame_0001.png" → "frame_0001.png.json".
        ann_path = ann_dir / f"{img_path.name}.json"
        if not ann_path.exists():
            # Also try stem-only: "frame_0001.json"
            ann_path = ann_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            skipped["missing-annotation"] += 1
            continue
        frames.append(BstldFrame(image_path=img_path.resolve(), ann_path=ann_path.resolve()))

    return frames, skipped


def _parse_supervisely_annotation(ann_path: Path) -> Tuple[int, int, List[Dict[str, Any]], str]:
    """Read a Supervisely annotation JSON and return (width, height, objects, error).

    On any parse failure *error* is a non-empty string and the other values
    should be ignored.
    """
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return 0, 0, [], f"json-error:{exc}"

    size = data.get("size", {})
    try:
        w = int(size["width"])
        h = int(size["height"])
    except (KeyError, TypeError, ValueError):
        return 0, 0, [], "missing-or-invalid-size"

    if w <= 0 or h <= 0:
        return 0, 0, [], "zero-size"

    objects = data.get("objects", []) or []
    return w, h, objects, ""


def _convert_bstld_frame(
    frame: BstldFrame,
    cfg: PipelineConfig,
    dst_images: Path,
    dst_labels: Path,
    class_counts: Counter,
    dropped: Counter,
) -> None:
    """Copy one BSTLD frame and write its YOLO label file."""
    ann_w, ann_h, objects, err = _parse_supervisely_annotation(frame.ann_path)
    if err:
        dropped[err] += 1
        return

    img = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    if img is None:
        dropped["unreadable-image"] += 1
        return

    image_h, image_w = img.shape[:2]

    # Unique output stem: prefix with parent folder name to avoid collisions
    # across sub-directories with identical frame filenames.
    parent_name = frame.image_path.parent.name
    out_stem = f"{parent_name}__{frame.image_path.stem}"
    out_image = dst_images / f"{out_stem}.jpg"
    out_label = dst_labels / f"{out_stem}.txt"

    labels: List[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            dropped["non-dict-object"] += 1
            continue

        raw_label = str(obj.get("classTitle", "")).strip()
        unified_name = BSTLD_LABEL_MAP.get(raw_label)
        if unified_name is None:
            dropped[f"unmapped-label:{raw_label}"] += 1
            continue

        class_id = BSTLD_CLASS_ID.get(unified_name)
        if class_id is None:
            dropped[f"missing-class:{unified_name}"] += 1
            continue

        # Supervisely bounding boxes: two exterior points [[x1,y1],[x2,y2]].
        try:
            exterior = obj["points"]["exterior"]
            (x1, y1), (x2, y2) = exterior[0], exterior[1]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
        except (KeyError, IndexError, TypeError, ValueError):
            dropped["invalid-bbox-points"] += 1
            continue

        yolo_bbox = xyxy_to_yolo(x1, y1, x2, y2, image_w, image_h, cfg.min_bbox_pixels)
        if yolo_bbox is None:
            dropped["bbox-too-small"] += 1
            continue

        labels.append(format_yolo_line(class_id, yolo_bbox))
        class_counts[class_id] += 1

    cv2.imwrite(str(out_image), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    write_label_file(out_label, labels, cfg.keep_empty_labels)


def prepare_iter_bstld_dataset(cfg: PipelineConfig, rng: random.Random) -> Dict[str, Any]:
    """Convert the Bosch Small Traffic Light Dataset (Supervisely export) to YOLO.

    Expected layout under ``cfg.bstld_root``::

        bstld/
          train/
            img/      ← training images
            ann/      ← Supervisely JSON annotations (one per image)
          test/
            img/
            ann/
          meta.json

    Produces a fully standalone dataset with its own 4-class taxonomy
    (red / yellow / green / off) defined by BSTLD_CLASS_NAMES — completely
    independent of the sign manifest used by the other pipeline stages.

    Training frames are split into *train* and *val* at the sub-directory
    level inside ``train/img/`` so no sequence straddles both splits.
    The ``test/`` split is written to ``images/test``.
    """
    dataset_dir = cfg.output_root / "iter_bstld"
    prepare_dataset_dir(dataset_dir, cfg.overwrite_output)
    ensure_dataset_layout(dataset_dir, ("train", "val", "test"))

    # ------------------------------------------------------------------
    # Collect frames
    # ------------------------------------------------------------------
    print("[iter_bstld] Collecting training frames …")
    train_frames, train_collect_skipped = _collect_bstld_frames(
        cfg.bstld_train_img_dir, cfg.bstld_train_ann_dir
    )

    print("[iter_bstld] Collecting test frames …")
    test_frames, test_collect_skipped = _collect_bstld_frames(
        cfg.bstld_test_img_dir, cfg.bstld_test_ann_dir
    )

    # ------------------------------------------------------------------
    # Stratified train / val split at the sub-directory (sequence) level
    # ------------------------------------------------------------------
    seq_to_frames: Dict[str, List[BstldFrame]] = defaultdict(list)
    for frame in train_frames:
        rel = frame.image_path.relative_to(cfg.bstld_train_img_dir.resolve())
        seq_key = rel.parts[0] if len(rel.parts) > 1 else frame.image_path.stem
        seq_to_frames[seq_key].append(frame)

    all_seqs = sorted(seq_to_frames.keys())
    rng.shuffle(all_seqs)
    n_val_seqs = max(1, int(len(all_seqs) * cfg.bstld_val_ratio))
    val_seqs: Set[str] = set(all_seqs[:n_val_seqs])

    split_train_frames = [f for seq, frames in seq_to_frames.items() if seq not in val_seqs for f in frames]
    split_val_frames   = [f for seq, frames in seq_to_frames.items() if seq in val_seqs     for f in frames]

    # ------------------------------------------------------------------
    # Convert each split
    # ------------------------------------------------------------------
    class_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    dropped:      Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}

    split_frame_map: Dict[str, List[BstldFrame]] = {
        "train": split_train_frames,
        "val":   split_val_frames,
        "test":  test_frames,
    }

    for split, frames in split_frame_map.items():
        print(f"[iter_bstld] Converting {split} ({len(frames)} frames) …")
        dst_images = dataset_dir / "images" / split
        dst_labels = dataset_dir / "labels" / split
        for frame in frames:
            _convert_bstld_frame(
                frame=frame,
                cfg=cfg,
                dst_images=dst_images,
                dst_labels=dst_labels,
                class_counts=class_counts[split],
                dropped=dropped[split],
            )

    # ------------------------------------------------------------------
    # Hard negative mining: copy road-scene images with no traffic lights
    # from iter_mapillary_backbone/images/train/ so the model learns what
    # traffic lights are *not*.  Skipped gracefully if mapillary hasn't
    # been generated or negatives are disabled (bstld_negatives=0).
    # ------------------------------------------------------------------
    n_negatives_added = 0
    negatives_skipped = ""
    if cfg.bstld_negatives > 0:
        mapillary_img_dir = cfg.output_root / "iter_mapillary_backbone" / "images" / "train"
        mapillary_lbl_dir = cfg.output_root / "iter_mapillary_backbone" / "labels" / "train"
        if not mapillary_img_dir.exists():
            negatives_skipped = "iter_mapillary_backbone not found — skipping hard negatives"
            print(f"[iter_bstld] WARNING: {negatives_skipped}")
        else:
            candidates = [
                p for p in sorted(mapillary_img_dir.iterdir())
                if p.is_file()
            ]
            # Only use images whose mapillary label file is empty (pure backgrounds),
            # to avoid accidentally importing sign annotations with wrong class IDs.
            empty_candidates = []
            for img_path in candidates:
                lbl_path = mapillary_lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                    empty_candidates.append(img_path)

            n_to_sample = min(cfg.bstld_negatives, len(empty_candidates))
            sampled = rng.sample(empty_candidates, n_to_sample)

            dst_neg_imgs = dataset_dir / "images" / "train"
            dst_neg_lbls = dataset_dir / "labels" / "train"
            for img_path in sampled:
                dst_img = dst_neg_imgs / f"neg__{img_path.name}"
                dst_lbl = dst_neg_lbls / f"neg__{img_path.stem}.txt"
                shutil.copy2(img_path, dst_img)
                dst_lbl.write_text("", encoding="utf-8")  # empty = no objects
                n_negatives_added += 1

            print(f"[iter_bstld] Added {n_negatives_added} hard negative images from Mapillary "
                  f"({len(empty_candidates)} empty candidates available)")

    yaml_path = write_dataset_yaml(dataset_dir, BSTLD_CLASS_NAMES, include_test=True)

    total_train = len(list((dataset_dir / "images" / "train").iterdir()))
    total_val   = len(list((dataset_dir / "images" / "val").iterdir()))
    total_test  = len(list((dataset_dir / "images" / "test").iterdir()))

    print(
        f"[iter_bstld] done — train={total_train}  val={total_val}  test={total_test}  "
        f"sequences_total={len(all_seqs)}  sequences_val={n_val_seqs}"
    )

    return {
        "dataset_dir": str(dataset_dir),
        "yaml_path": str(yaml_path),
        "validate_splits": ["train", "val", "test"],
        "report": {
            "sequences_total": len(all_seqs),
            "sequences_val": n_val_seqs,
            "split_images": {"train": total_train, "val": total_val, "test": total_test},
            "class_counts": {s: counter_to_dict(c) for s, c in class_counts.items()},
            "dropped": {s: counter_to_dict(c) for s, c in dropped.items()},
            "collect_skipped": {
                "train": counter_to_dict(train_collect_skipped),
                "test":  counter_to_dict(test_collect_skipped),
            },
            "hard_negatives_added": n_negatives_added,
            "hard_negatives_skipped": negatives_skipped,
        },
    }


# ---------------------------------------------------------------------------
# Romanian dataset support
# ---------------------------------------------------------------------------

def build_romanian_class_map(manifest: ClassManifest) -> Tuple[Dict[int, Optional[int]], List[Dict[str, object]]]:
    base_id_by_name = {name: idx for idx, name in enumerate(BASE_CLASS_NAMES)}
    class_map: Dict[int, Optional[int]] = {}
    map_table: List[Dict[str, object]] = []

    for source_id, source_name in enumerate(manifest.romanian_source_names):
        if source_name in ROMANIAN_DROP_CLASS_NAMES:
            class_map[source_id] = None
            map_table.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "action": "drop",
                    "target_id": None,
                    "target_name": None,
                    "reason": "unsupported-speed-policy",
                }
            )
            continue

        if source_name in ROMANIAN_FORCE_APPEND_CLASS_NAMES:
            target_id = manifest.id_by_name.get(source_name)
            if target_id is None:
                raise KeyError(f"Forced Romanian class '{source_name}' is missing from unified class manifest.")

            class_map[source_id] = target_id
            map_table.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "action": "force-append",
                    "target_id": target_id,
                    "target_name": source_name,
                    "reason": "explicit-force-append-policy",
                }
            )
            continue

        mapped_base = ROMANIAN_TO_BASE_CLASS.get(source_name)
        if mapped_base is not None:
            target_id = base_id_by_name[mapped_base]
            class_map[source_id] = target_id
            map_table.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "action": "map-existing",
                    "target_id": target_id,
                    "target_name": mapped_base,
                    "reason": "conservative-semantic-match",
                }
            )
            continue

        if source_name in base_id_by_name:
            target_id = base_id_by_name[source_name]
            class_map[source_id] = target_id
            map_table.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "action": "map-existing",
                    "target_id": target_id,
                    "target_name": source_name,
                    "reason": "exact-base-name-match",
                }
            )
            continue

        target_id = manifest.id_by_name.get(source_name)
        if target_id is None:
            raise KeyError(f"Romanian class '{source_name}' missing from unified class manifest.")

        class_map[source_id] = target_id
        map_table.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "action": "append-new",
                "target_id": target_id,
                "target_name": source_name,
                "reason": "no-clean-match-in-base-43",
            }
        )

    return class_map, map_table


# ---------------------------------------------------------------------------
# Mapillary dataset support
# ---------------------------------------------------------------------------

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


def prepare_iter_mapillary_backbone_dataset(cfg: PipelineConfig, manifest: ClassManifest) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# GTSDB + synthetic dataset support
# ---------------------------------------------------------------------------

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
        min(background_w, background_h) * rng.uniform(cfg.synthetic_min_rel_size, cfg.synthetic_max_rel_size)
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


# ---------------------------------------------------------------------------
# Romanian dataset support (conversion)
# ---------------------------------------------------------------------------

def find_split_root(source_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        split_root = source_root / candidate
        if split_root.exists() and split_root.is_dir():
            return split_root
    return None


def convert_romanian_split(
    split_name: str,
    source_split_root: Path,
    output_root: Path,
    class_map: Dict[int, Optional[int]],
    source_names: List[str],
    source_counts: Dict[str, Counter],
    target_counts: Dict[str, Counter],
    dropped_by_source_name: Counter,
    keep_empty_labels: bool,
) -> Dict[str, int]:
    source_images = source_split_root / "images"
    source_labels = source_split_root / "labels"
    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"Missing images/labels folders under: {source_split_root}")

    out_images = output_root / "images" / split_name
    out_labels = output_root / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(path for path in source_images.iterdir() if path.is_file())
    stats = {
        "images": 0,
        "missing_label_files": 0,
        "kept_instances": 0,
        "dropped_instances": 0,
        "invalid_label_lines": 0,
        "unknown_source_ids": 0,
    }

    for source_image in image_files:
        destination_image = out_images / source_image.name
        destination_label = out_labels / f"{source_image.stem}.txt"
        source_label = source_labels / f"{source_image.stem}.txt"

        shutil.copy2(source_image, destination_image)
        stats["images"] += 1

        if not source_label.exists():
            stats["missing_label_files"] += 1
            write_label_file(destination_label, [], keep_empty_labels)
            continue

        converted_lines: List[str] = []
        for raw_line in source_label.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue

            parsed = parse_yolo_line_payload(raw_line)
            if parsed is None:
                stats["invalid_label_lines"] += 1
                continue

            source_id, box_payload = parsed
            if source_id < 0 or source_id >= len(source_names):
                stats["unknown_source_ids"] += 1
                continue

            source_name = source_names[source_id]
            source_counts[split_name][source_name] += 1
            target_id = class_map.get(source_id)
            if target_id is None:
                stats["dropped_instances"] += 1
                dropped_by_source_name[source_name] += 1
                continue

            converted_lines.append(f"{target_id} {box_payload}")
            stats["kept_instances"] += 1
            target_counts[split_name][target_id] += 1

        write_label_file(destination_label, converted_lines, keep_empty_labels)

    return stats


def prepare_iter_romanian_dataset(cfg: PipelineConfig, manifest: ClassManifest) -> Dict[str, Any]:
    dataset_dir = cfg.output_root / "iter_romanian"
    prepare_dataset_dir(dataset_dir, cfg.overwrite_output)

    class_map, map_table = build_romanian_class_map(manifest)
    source_root = cfg.romanian_root

    train_root = find_split_root(source_root, ("train",))
    val_root = find_split_root(source_root, ("valid", "val"))
    test_root = find_split_root(source_root, ("test",))
    if train_root is None or val_root is None:
        raise FileNotFoundError("Expected train and valid/val directories under Romanian source-root.")

    source_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    target_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    dropped_by_source_name: Counter = Counter()

    train_stats = convert_romanian_split(
        split_name="train",
        source_split_root=train_root,
        output_root=dataset_dir,
        class_map=class_map,
        source_names=manifest.romanian_source_names,
        source_counts=source_counts,
        target_counts=target_counts,
        dropped_by_source_name=dropped_by_source_name,
        keep_empty_labels=cfg.keep_empty_labels,
    )

    val_stats = convert_romanian_split(
        split_name="val",
        source_split_root=val_root,
        output_root=dataset_dir,
        class_map=class_map,
        source_names=manifest.romanian_source_names,
        source_counts=source_counts,
        target_counts=target_counts,
        dropped_by_source_name=dropped_by_source_name,
        keep_empty_labels=cfg.keep_empty_labels,
    )

    include_test = False
    test_stats: Optional[Dict[str, int]] = None
    if test_root is not None and (test_root / "images").exists() and (test_root / "labels").exists():
        source_counts["test"] = Counter()
        target_counts["test"] = Counter()
        test_stats = convert_romanian_split(
            split_name="test",
            source_split_root=test_root,
            output_root=dataset_dir,
            class_map=class_map,
            source_names=manifest.romanian_source_names,
            source_counts=source_counts,
            target_counts=target_counts,
            dropped_by_source_name=dropped_by_source_name,
            keep_empty_labels=cfg.keep_empty_labels,
        )
        include_test = True

    yaml_path = write_dataset_yaml(dataset_dir, manifest.names, include_test=include_test)

    reports_dir = dataset_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    map_json_path = reports_dir / "class_mapping.json"
    map_json_path.write_text(json.dumps(map_table, indent=2), encoding="utf-8")

    appended_classes = [
        {
            "target_id": row["target_id"],
            "target_name": row["target_name"],
            "source_id": row["source_id"],
            "source_name": row["source_name"],
        }
        for row in map_table
        if row["action"] in ("append-new", "force-append")
    ]

    missing_force_append_classes = sorted(
        ROMANIAN_FORCE_APPEND_CLASS_NAMES - set(manifest.romanian_source_names)
    )

    remap_summary = {
        "source_root": str(source_root.resolve()),
        "output_root": str(dataset_dir.resolve()),
        "base_nc": len(BASE_CLASS_NAMES),
        "target_nc": len(manifest.names),
        "source_nc": len(manifest.romanian_source_names),
        "source_names": manifest.romanian_source_names,
        "target_names": manifest.names,
        "mapping_policy": {
            "unsupported_speed_classes_dropped": sorted(ROMANIAN_DROP_CLASS_NAMES),
            "conservative_existing_map_count": len(ROMANIAN_TO_BASE_CLASS),
            "append_new_classes": True,
            "force_append_classes": sorted(ROMANIAN_FORCE_APPEND_CLASS_NAMES),
            "missing_force_append_classes_in_source": missing_force_append_classes,
            "keep_base_ids_0_42": True,
        },
        "stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "source_instance_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in source_counts.items()
        },
        "target_instance_counts": {
            split: {str(k): int(v) for k, v in sorted(counter.items())}
            for split, counter in target_counts.items()
        },
        "dropped_instances_by_source_class": dict(sorted(dropped_by_source_name.items())),
        "appended_classes": appended_classes,
        "artifacts": {
            "dataset_yaml": str(yaml_path),
            "class_mapping_json": str(map_json_path),
        },
    }
    remap_summary_path = reports_dir / "remap_summary.json"
    remap_summary_path.write_text(json.dumps(remap_summary, indent=2), encoding="utf-8")

    print(
        "[iter_romanian] train_images=",
        int(train_stats["images"]),
        "val_images=",
        int(val_stats["images"]),
        "target_nc=",
        len(manifest.names),
    )

    validate_splits = ["train", "val"]
    if include_test:
        validate_splits.append("test")

    return {
        "dataset_dir": str(dataset_dir),
        "yaml_path": str(yaml_path),
        "validate_splits": validate_splits,
        "report": {
            "train_stats": train_stats,
            "val_stats": val_stats,
            "test_stats": test_stats,
            "target_nc": len(manifest.names),
            "class_mapping_json": str(map_json_path),
            "remap_summary_json": str(remap_summary_path),
            "appended_classes": appended_classes,
        },
    }


# ---------------------------------------------------------------------------
# Validation helpers
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

        image_files = sorted(path for path in images_dir.iterdir() if path.is_file()) if images_dir.exists() else []
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


def existing_iterative_dataset_is_usable(dataset_dir: Path, required_splits: Sequence[str]) -> bool:
    if not dataset_dir.exists() or not (dataset_dir / "dataset.yaml").exists():
        return False

    for split in required_splits:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            return False

    return True


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig, requested_datasets: Sequence[str]) -> None:
    prepare_output_root(cfg)

    requested = list(requested_datasets)
    execution_targets, auto_added = expand_target_dependencies(requested)
    auto_added_set: Set[str] = set(auto_added)

    print("Preparing output directory:", cfg.output_root)
    print("Requested datasets:", ", ".join(requested))
    if auto_added:
        print("Auto-added dependencies:", ", ".join(auto_added))
    print("Execution order:", ", ".join(execution_targets))

    reports_dir = cfg.output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # The sign manifest (reads romanian data.yaml) is only needed for the
    # three sign-dataset stages.  Skip it entirely when running bstld-only
    # so the script works without the Romanian dataset being present.
    sign_targets = {"iter_mapillary_backbone", "iter_gtsdb_plus_synthetic", "iter_romanian"}
    needs_manifest = bool(sign_targets & set(execution_targets))
    manifest: Optional[ClassManifest] = None
    class_manifest_path: Optional[Path] = None
    if needs_manifest:
        manifest = build_class_manifest(cfg)
        class_manifest_path = reports_dir / "class_manifest.json"
        write_json(class_manifest_path, class_manifest_to_dict(manifest))

    rng = random.Random(cfg.seed)
    target_outputs: Dict[str, Dict[str, Any]] = {}
    train_backgrounds: List[Tuple[Path, Path]] = []

    if "iter_mapillary_backbone" in execution_targets:
        existing_mapillary_dir = cfg.output_root / "iter_mapillary_backbone"
        if (
            "iter_mapillary_backbone" in auto_added_set
            and existing_iterative_dataset_is_usable(existing_mapillary_dir, ("train", "val", "test"))
        ):
            print("\n[Skip] Reusing existing iter_mapillary_backbone.")
            train_backgrounds = collect_train_backgrounds(existing_mapillary_dir)
        else:
            print("\n[Run] Building iter_mapillary_backbone...")
            mapillary_output = prepare_iter_mapillary_backbone_dataset(cfg, manifest)  # type: ignore[arg-type]
            target_outputs["iter_mapillary_backbone"] = mapillary_output
            train_backgrounds = mapillary_output["train_backgrounds"]

    if "iter_gtsdb_plus_synthetic" in execution_targets:
        existing_gtsdb_dir = cfg.output_root / "iter_gtsdb_plus_synthetic"
        if (
            "iter_gtsdb_plus_synthetic" in auto_added_set
            and existing_iterative_dataset_is_usable(existing_gtsdb_dir, ("train", "val"))
        ):
            print("\n[Skip] Reusing existing iter_gtsdb_plus_synthetic.")
        else:
            if not train_backgrounds:
                existing_mapillary_dir = cfg.output_root / "iter_mapillary_backbone"
                train_backgrounds = collect_train_backgrounds(existing_mapillary_dir)
            if not train_backgrounds:
                raise FileNotFoundError(
                    "No Mapillary train backgrounds available. Generate iter_mapillary_backbone first."
                )

            print("\n[Run] Building iter_gtsdb_plus_synthetic...")
            gtsdb_output = prepare_iter_gtsdb_plus_synthetic_dataset(cfg, manifest, rng, train_backgrounds)  # type: ignore[arg-type]
            target_outputs["iter_gtsdb_plus_synthetic"] = gtsdb_output

    if "iter_romanian" in execution_targets:
        existing_romanian_dir = cfg.output_root / "iter_romanian"
        if (
            "iter_romanian" in auto_added_set
            and existing_iterative_dataset_is_usable(existing_romanian_dir, ("train", "val"))
        ):
            print("\n[Skip] Reusing existing iter_romanian.")
        else:
            print("\n[Run] Building iter_romanian...")
            romanian_output = prepare_iter_romanian_dataset(cfg, manifest)  # type: ignore[arg-type]
            target_outputs["iter_romanian"] = romanian_output

    if "iter_bstld" in execution_targets:
        existing_bstld_dir = cfg.output_root / "iter_bstld"
        if (
            "iter_bstld" in auto_added_set
            and existing_iterative_dataset_is_usable(existing_bstld_dir, ("train", "val", "test"))
        ):
            print("\n[Skip] Reusing existing iter_bstld.")
        else:
            print("\n[Run] Building iter_bstld...")
            bstld_output = prepare_iter_bstld_dataset(cfg, rng)
            target_outputs["iter_bstld"] = bstld_output

    print("\n[Run] Validating generated targets...")
    validation: Dict[str, Any] = {}
    for target in execution_targets:
        output = target_outputs.get(target)
        if output is None:
            continue
        if target == "iter_bstld":
            class_names = BSTLD_CLASS_NAMES
        else:
            class_names = manifest.names  # type: ignore[union-attr]
        validation[target] = validate_dataset(
            dataset_dir=Path(output["dataset_dir"]),
            splits=output["validate_splits"],
            class_names=class_names,
        )

    artifacts: Dict[str, Any] = {}
    if class_manifest_path is not None:
        artifacts["class_manifest_json"] = str(class_manifest_path)
    target_reports: Dict[str, Any] = {}
    for target, output in target_outputs.items():
        artifacts[f"{target}_yaml"] = str(output["yaml_path"])
        target_reports[target] = output["report"]

    summary: Dict[str, Any] = {
        "seed": cfg.seed,
        "requested_datasets": requested,
        "execution_datasets": execution_targets,
        "auto_added_dependencies": auto_added,
        "targets": target_reports,
        "validation": validation,
        "artifacts": artifacts,
    }
    if manifest is not None:
        summary["class_manifest"] = class_manifest_to_dict(manifest)
    summary_path = reports_dir / "hybrid_pipeline_summary.json"
    write_json(summary_path, summary)

    print("\nDone. Output summary:", summary_path)
    for artifact_name, artifact_path in artifacts.items():
        print(f"{artifact_name}: {artifact_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_data_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build hybrid YOLO datasets with one unified class space across iterative stages."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma or space-separated targets to generate. "
            f"Allowed: all, {', '.join(DATASET_TARGETS)}. "
            "Aliases: mapillary, gtsdb, romanian, bstld"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Root data directory (default: script directory)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_data_root / "traffic_sign_pipeline",
        help="Output directory for generated datasets and reports",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits and synthesis")
    parser.add_argument(
        "--gtsdb-val-ratio",
        type=float,
        default=0.20,
        help="Fraction of GTSDB images reserved for validation (default: 0.20)",
    )
    parser.add_argument(
        "--bstld-val-ratio",
        type=float,
        default=0.15,
        help="Fraction of BSTLD training bags reserved for validation (default: 0.15)",
    )
    parser.add_argument(
        "--bstld-negatives",
        type=int,
        default=2000,
        help="Number of empty Mapillary frames to add as hard negatives to iter_bstld train split (default: 2000, 0 to disable)",
    )
    parser.add_argument(
        "--synthetic-min-instances",
        type=int,
        default=250,
        help="Minimum target count per class when computing synthetic deficits",
    )
    parser.add_argument(
        "--synthetic-max-per-class",
        type=int,
        default=400,
        help="Cap of generated synthetic samples per class",
    )
    parser.add_argument(
        "--exclude-gtsrb-test",
        action="store_true",
        help="Use only GTSRB Train.csv as crop source (exclude Test.csv)",
    )
    parser.add_argument(
        "--no-overwrite-output",
        action="store_true",
        help="Fail if a selected output dataset already exists instead of replacing it",
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if not 0.0 < float(args.gtsdb_val_ratio) < 1.0:
        raise ValueError("--gtsdb-val-ratio must be between 0 and 1")
    if not 0.0 < float(args.bstld_val_ratio) < 1.0:
        raise ValueError("--bstld-val-ratio must be between 0 and 1")
    if args.synthetic_min_instances <= 0:
        raise ValueError("--synthetic-min-instances must be > 0")
    if args.synthetic_max_per_class <= 0:
        raise ValueError("--synthetic-max-per-class must be > 0")

    return PipelineConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        seed=args.seed,
        gtsdb_val_ratio=args.gtsdb_val_ratio,
        bstld_val_ratio=args.bstld_val_ratio,
        bstld_negatives=args.bstld_negatives,
        overwrite_output=not args.no_overwrite_output,
        include_gtsrb_test=not args.exclude_gtsrb_test,
        filter_occluded_bstld=not args.no_filter_occluded_bstld,
        synthetic_min_instances=args.synthetic_min_instances,
        synthetic_max_per_class=args.synthetic_max_per_class,
    )


def main() -> None:
    args = parse_args()
    requested_datasets = parse_requested_datasets(args.datasets)
    config = build_config_from_args(args)
    run_pipeline(config, requested_datasets)


if __name__ == "__main__":
    main()