from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import yaml


# Canonical GTSDB/GTSRB 43-class taxonomy.
GTSDB_CLASS_NAMES: List[str] = [
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


# Prefix-based mapping from Mapillary labels to the 43-class GTSDB taxonomy.
# This is intentionally conservative: labels that do not map clearly are dropped.
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


@dataclass(frozen=True)
class CropSample:
	image_path: Path
	class_id: int
	x1: int
	y1: int
	x2: int
	y2: int


@dataclass
class PipelineConfig:
	data_root: Path
	output_root: Path
	seed: int = 42
	gtsdb_val_ratio: float = 0.20
	overwrite_output: bool = True
	keep_empty_labels: bool = True
	include_gtsrb_test: bool = True
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
	def mapillary_root(self) -> Path:
		return self.data_root / "mapillary"

	@property
	def mapillary_annotations_dir(self) -> Path:
		return self.mapillary_root / "annotations"

	@property
	def mapillary_splits_dir(self) -> Path:
		return self.mapillary_root / "splits"


def clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(value, upper))


def counter_to_dict(counter: Counter) -> Dict[str, int]:
	return {str(k): int(counter[k]) for k in sorted(counter)}


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


def format_yolo_line(class_id: int, bbox: Tuple[float, float, float, float]) -> str:
	x_center, y_center, width, height = bbox
	return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


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


def load_split_keys(split_file: Path) -> List[str]:
	if not split_file.exists():
		return []
	return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


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

	# Lightweight fallback mappings for frequent variants.
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


def is_usable_mapillary_object(obj: Dict, cfg: PipelineConfig) -> Tuple[bool, str]:
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


def write_dataset_yaml(dataset_dir: Path, include_test: bool) -> Path:
	data: Dict[str, object] = {
		"path": str(dataset_dir.resolve()),
		"train": "images/train",
		"val": "images/val",
		"nc": len(GTSDB_CLASS_NAMES),
		"names": GTSDB_CLASS_NAMES,
	}
	if include_test:
		data["test"] = "images/test"

	yaml_path = dataset_dir / "dataset.yaml"
	with yaml_path.open("w", encoding="utf-8") as stream:
		yaml.safe_dump(data, stream, sort_keys=False)
	return yaml_path


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

			if class_id < 0 or class_id >= len(GTSDB_CLASS_NAMES):
				continue
			grouped[image_name].append((class_id, x1, y1, x2, y2))
	return grouped


def prepare_mapillary_backbone_dataset(cfg: PipelineConfig) -> Dict[str, Any]:
	dataset_dir = cfg.output_root / "mapillary_backbone"
	dataset_dir.mkdir(parents=True, exist_ok=True)

	class_counts: Counter = Counter()
	dropped: Counter = Counter()
	train_backgrounds: List[Tuple[Path, Path]] = []
	split_roots: Dict[str, Tuple[Path, Path]] = {}
	processed_images = 0
	kept_objects = 0

	for split in ("train", "val", "test"):
		images_dir = cfg.mapillary_root / split / "images"
		labels_dir = cfg.mapillary_root / split / "labels"
		labels_dir.mkdir(parents=True, exist_ok=True)
		split_roots[split] = (images_dir, labels_dir)

		split_keys = load_split_keys(cfg.mapillary_splits_dir / f"{split}.txt")
		for key in split_keys:
			source_image = images_dir / f"{key}.jpg"
			source_annotation = cfg.mapillary_annotations_dir / f"{key}.json"
			if not source_image.exists() or not source_annotation.exists():
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

			stem = key
			destination_label = labels_dir / f"{stem}.txt"
			write_label_file(destination_label, labels, cfg.keep_empty_labels)

			if split == "train":
				train_backgrounds.append((source_image, destination_label))

			processed_images += 1

	yaml_path = dataset_dir / "dataset.yaml"
	stage1_yaml = {
		"path": str(cfg.mapillary_root.resolve()),
		"train": str((cfg.mapillary_root / "train" / "images").resolve()),
		"val": str((cfg.mapillary_root / "val" / "images").resolve()),
		"test": str((cfg.mapillary_root / "test" / "images").resolve()),
		"nc": len(GTSDB_CLASS_NAMES),
		"names": GTSDB_CLASS_NAMES,
	}
	with yaml_path.open("w", encoding="utf-8") as stream:
		yaml.safe_dump(stage1_yaml, stream, sort_keys=False)

	print(
		"[Mapillary] images=",
		processed_images,
		"kept_objects=",
		kept_objects,
		"dropped_objects=",
		int(sum(dropped.values())),
	)

	return {
		"dataset_dir": dataset_dir,
		"yaml_path": yaml_path,
		"split_roots": split_roots,
		"train_backgrounds": train_backgrounds,
		"class_counts": class_counts,
		"dropped": dropped,
		"processed_images": processed_images,
		"kept_objects": kept_objects,
	}


def prepare_gtsdb_finetune_dataset(cfg: PipelineConfig, rng: random.Random) -> Dict[str, Any]:
	dataset_dir = cfg.output_root / "gtsdb_finetune"
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

	yaml_path = write_dataset_yaml(dataset_dir, include_test=False)
	print(
		"[GTSDB] images=",
		processed_images,
		"train_objects=",
		int(sum(class_counts_train.values())),
		"val_objects=",
		int(sum(class_counts_val.values())),
	)

	return {
		"dataset_dir": dataset_dir,
		"yaml_path": yaml_path,
		"class_counts_train": class_counts_train,
		"class_counts_val": class_counts_val,
		"processed_images": processed_images,
	}


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
			if class_id < 0 or class_id >= len(GTSDB_CLASS_NAMES):
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

	target_max_side = int(min(background_w, background_h) * rng.uniform(cfg.synthetic_min_rel_size, cfg.synthetic_max_rel_size))
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
	mapillary_stats: Dict[str, Any],
	gtsdb_stats: Dict[str, Any],
	crop_bank: Dict[int, List[CropSample]],
) -> Dict[str, Any]:
	dataset_dir = cfg.output_root / "synthetic_pool"
	ensure_dataset_layout(dataset_dir, ("train",))

	train_backgrounds: List[Tuple[Path, Path]] = mapillary_stats["train_backgrounds"]  # type: ignore[index]
	train_counts: Counter = gtsdb_stats["class_counts_train"]  # type: ignore[index]

	nonzero_counts = [count for count in train_counts.values() if count > 0]
	median_count = int(statistics.median(nonzero_counts)) if nonzero_counts else cfg.synthetic_min_instances
	target_count = max(cfg.synthetic_min_instances, median_count)

	planned_per_class: Dict[int, int] = {}
	for class_id in range(len(GTSDB_CLASS_NAMES)):
		if not crop_bank.get(class_id):
			continue
		deficit = target_count - int(train_counts.get(class_id, 0))
		if deficit > 0:
			planned_per_class[class_id] = min(deficit, cfg.synthetic_max_per_class)

	generated_per_class: Counter = Counter()
	failures: Counter = Counter()
	image_index = 0

	if not train_backgrounds:
		print("[Synthetic] No Mapillary train backgrounds found. Skipping synthesis.")
		return {
			"dataset_dir": dataset_dir,
			"planned_per_class": planned_per_class,
			"generated_per_class": generated_per_class,
			"failures": failures,
			"target_count": target_count,
			"generated_images": image_index,
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

	print(
		"[Synthetic] generated_images=",
		image_index,
		"target_count=",
		target_count,
		"planned=",
		int(sum(planned_per_class.values())),
		"generated=",
		int(sum(generated_per_class.values())),
	)

	return {
		"dataset_dir": dataset_dir,
		"planned_per_class": planned_per_class,
		"generated_per_class": generated_per_class,
		"failures": failures,
		"target_count": target_count,
		"generated_images": image_index,
	}


def copy_split(src_dataset: Path, dst_dataset: Path, split: str, keep_empty_labels: bool) -> int:
	src_images = src_dataset / "images" / split
	src_labels = src_dataset / "labels" / split
	dst_images = dst_dataset / "images" / split
	dst_labels = dst_dataset / "labels" / split
	if not src_images.exists():
		return 0

	copied = 0
	for image_file in sorted(src_images.iterdir()):
		if not image_file.is_file():
			continue

		destination_image = dst_images / image_file.name
		destination_label = dst_labels / f"{image_file.stem}.txt"
		source_label = src_labels / f"{image_file.stem}.txt"

		shutil.copy2(image_file, destination_image)
		if source_label.exists():
			shutil.copy2(source_label, destination_label)
		else:
			write_label_file(destination_label, [], keep_empty_labels)

		copied += 1
	return copied


def assemble_gtsdb_plus_synthetic_dataset(
	cfg: PipelineConfig,
	gtsdb_dataset_dir: Path,
	synthetic_dataset_dir: Path,
) -> Dict[str, Any]:
	dataset_dir = cfg.output_root / "gtsdb_plus_synthetic"
	ensure_dataset_layout(dataset_dir, ("train", "val"))

	copied_train_gtsdb = copy_split(gtsdb_dataset_dir, dataset_dir, "train", cfg.keep_empty_labels)
	copied_val_gtsdb = copy_split(gtsdb_dataset_dir, dataset_dir, "val", cfg.keep_empty_labels)
	copied_train_synthetic = copy_split(synthetic_dataset_dir, dataset_dir, "train", cfg.keep_empty_labels)

	yaml_path = write_dataset_yaml(dataset_dir, include_test=False)
	print(
		"[Stage3] train_gtsdb=",
		copied_train_gtsdb,
		"train_synthetic=",
		copied_train_synthetic,
		"val_gtsdb=",
		copied_val_gtsdb,
	)

	return {
		"dataset_dir": dataset_dir,
		"yaml_path": yaml_path,
		"copied_train_gtsdb": copied_train_gtsdb,
		"copied_train_synthetic": copied_train_synthetic,
		"copied_val_gtsdb": copied_val_gtsdb,
	}


def validate_dataset(dataset_dir: Path, splits: Sequence[str]) -> Dict[str, Any]:
	split_reports: Dict[str, Any] = {}
	total_missing_labels = 0
	total_invalid_lines = 0

	for split in splits:
		images_dir = dataset_dir / "images" / split
		labels_dir = dataset_dir / "labels" / split

		image_files = sorted([path for path in images_dir.iterdir() if path.is_file()]) if images_dir.exists() else []
		missing_labels = 0
		invalid_lines = 0
		class_counts: Counter = Counter()

		for image_path in image_files:
			label_path = labels_dir / f"{image_path.stem}.txt"
			if not label_path.exists():
				missing_labels += 1
				continue

			lines = read_yolo_lines(label_path)
			for line in lines:
				parsed = parse_yolo_line(line)
				if parsed is None:
					invalid_lines += 1
					continue

				class_id, _, _, width, height = parsed
				if class_id < 0 or class_id >= len(GTSDB_CLASS_NAMES):
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


def validate_split_roots(split_roots: Dict[str, Tuple[Path, Path]], dataset_name: str) -> Dict[str, Any]:
	split_reports: Dict[str, Any] = {}
	total_missing_labels = 0
	total_invalid_lines = 0

	for split, roots in split_roots.items():
		images_dir, labels_dir = roots
		image_files = sorted([path for path in images_dir.iterdir() if path.is_file()]) if images_dir.exists() else []
		missing_labels = 0
		invalid_lines = 0
		class_counts: Counter = Counter()

		for image_path in image_files:
			label_path = labels_dir / f"{image_path.stem}.txt"
			if not label_path.exists():
				missing_labels += 1
				continue

			lines = read_yolo_lines(label_path)
			for line in lines:
				parsed = parse_yolo_line(line)
				if parsed is None:
					invalid_lines += 1
					continue

				class_id, _, _, width, height = parsed
				if class_id < 0 or class_id >= len(GTSDB_CLASS_NAMES):
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
			"images_dir": str(images_dir),
			"labels_dir": str(labels_dir),
			"missing_labels": missing_labels,
			"invalid_label_lines": invalid_lines,
			"class_counts": counter_to_dict(class_counts),
		}

	return {
		"dataset": dataset_name,
		"splits": split_reports,
		"total_missing_labels": total_missing_labels,
		"total_invalid_label_lines": total_invalid_lines,
	}


def prepare_output_root(cfg: PipelineConfig) -> None:
	output_root = cfg.output_root.resolve()
	data_root = cfg.data_root.resolve()
	if output_root == data_root:
		raise ValueError("output_root must not be the same as data_root")

	if output_root.exists():
		if not cfg.overwrite_output:
			raise FileExistsError(
				f"Output directory already exists: {output_root}. Use overwrite mode or remove it manually."
			)
		shutil.rmtree(output_root)

	output_root.mkdir(parents=True, exist_ok=True)


def run_pipeline(cfg: PipelineConfig) -> None:
	print("Preparing output directory:", cfg.output_root)
	prepare_output_root(cfg)

	rng = random.Random(cfg.seed)

	print("\n[1/6] Converting Mapillary backbone dataset...")
	mapillary_stats = prepare_mapillary_backbone_dataset(cfg)

	print("\n[2/6] Converting GTSDB fine-tune dataset...")
	gtsdb_stats = prepare_gtsdb_finetune_dataset(cfg, rng)

	print("\n[3/6] Building GTSRB crop bank...")
	crop_bank = build_gtsrb_crop_bank(cfg)

	print("\n[4/6] Generating synthetic samples for under-represented classes...")
	synthetic_stats = generate_synthetic_samples(cfg, rng, mapillary_stats, gtsdb_stats, crop_bank)

	print("\n[5/6] Assembling GTSDB + synthetic training dataset...")
	stage3_stats = assemble_gtsdb_plus_synthetic_dataset(
		cfg,
		gtsdb_dataset_dir=gtsdb_stats["dataset_dir"],  # type: ignore[index]
		synthetic_dataset_dir=synthetic_stats["dataset_dir"],  # type: ignore[index]
	)

	print("\n[6/6] Validating outputs and writing summary report...")
	validation_report = {
		"mapillary_backbone": validate_split_roots(mapillary_stats["split_roots"], "mapillary_backbone"),  # type: ignore[index]
		"gtsdb_finetune": validate_dataset(gtsdb_stats["dataset_dir"], ("train", "val")),  # type: ignore[index]
		"gtsdb_plus_synthetic": validate_dataset(stage3_stats["dataset_dir"], ("train", "val")),  # type: ignore[index]
	}

	summary = {
		"seed": cfg.seed,
		"class_names": GTSDB_CLASS_NAMES,
		"artifacts": {
			"mapillary_yaml": str(mapillary_stats["yaml_path"]),
			"gtsdb_yaml": str(gtsdb_stats["yaml_path"]),
			"gtsdb_plus_synthetic_yaml": str(stage3_stats["yaml_path"]),
		},
		"mapillary": {
			"processed_images": int(mapillary_stats["processed_images"]),
			"kept_objects": int(mapillary_stats["kept_objects"]),
			"class_counts": counter_to_dict(mapillary_stats["class_counts"]),
			"dropped_objects": counter_to_dict(mapillary_stats["dropped"]),
			"train_backgrounds": len(mapillary_stats["train_backgrounds"]),
			"labels_written_in_place": True,
			"split_roots": {
				k: {
					"images_dir": str(v[0]),
					"labels_dir": str(v[1]),
				}
				for k, v in mapillary_stats["split_roots"].items()
			},
		},
		"gtsdb": {
			"processed_images": int(gtsdb_stats["processed_images"]),
			"train_class_counts": counter_to_dict(gtsdb_stats["class_counts_train"]),
			"val_class_counts": counter_to_dict(gtsdb_stats["class_counts_val"]),
		},
		"synthetic": {
			"target_count": int(synthetic_stats["target_count"]),
			"planned_per_class": {str(k): int(v) for k, v in sorted(synthetic_stats["planned_per_class"].items())},
			"generated_per_class": counter_to_dict(synthetic_stats["generated_per_class"]),
			"failures": counter_to_dict(synthetic_stats["failures"]),
			"generated_images": int(synthetic_stats["generated_images"]),
		},
		"stage3": {
			"copied_train_gtsdb": int(stage3_stats["copied_train_gtsdb"]),
			"copied_train_synthetic": int(stage3_stats["copied_train_synthetic"]),
			"copied_val_gtsdb": int(stage3_stats["copied_val_gtsdb"]),
		},
		"validation": validation_report,
	}

	reports_dir = cfg.output_root / "reports"
	reports_dir.mkdir(parents=True, exist_ok=True)
	summary_path = reports_dir / "pipeline_summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print("\nDone. Output summary:", summary_path)
	print("Stage 1 YAML:", mapillary_stats["yaml_path"])
	print("Stage 2 YAML:", gtsdb_stats["yaml_path"])
	print("Stage 3 YAML:", stage3_stats["yaml_path"])


def parse_args() -> argparse.Namespace:
	default_data_root = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(
		description=(
			"Prepare YOLO datasets for staged traffic-sign training: "
			"Mapillary backbone, GTSDB fine-tune, and GTSRB-based synthetic augmentation."
		)
	)
	parser.add_argument("--data-root", type=Path, default=default_data_root, help="Root data directory (default: data/)")
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
		help="Fraction of GTSDB images used for validation split",
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
		help="Use only GTSRB Train.csv as crop source",
	)
	parser.add_argument(
		"--no-overwrite-output",
		action="store_true",
		help="Fail if output-root exists instead of deleting it",
	)
	return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
	if not 0.0 < float(args.gtsdb_val_ratio) < 1.0:
		raise ValueError("--gtsdb-val-ratio must be between 0 and 1")

	return PipelineConfig(
		data_root=args.data_root,
		output_root=args.output_root,
		seed=args.seed,
		gtsdb_val_ratio=args.gtsdb_val_ratio,
		overwrite_output=not args.no_overwrite_output,
		include_gtsrb_test=not args.exclude_gtsrb_test,
		synthetic_min_instances=args.synthetic_min_instances,
		synthetic_max_per_class=args.synthetic_max_per_class,
	)


def main() -> None:
	args = parse_args()
	config = build_config_from_args(args)
	run_pipeline(config)


if __name__ == "__main__":
	main()
