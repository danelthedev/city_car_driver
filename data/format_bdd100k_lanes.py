from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


# Output classes for YOLO segmentation.
# Keep fine-grained BDD lane/area subclasses so downstream logic can
# distinguish lane types when demarcating lanes and intersections.
LANE_CLASS_NAMES: List[str] = [
    "area_drivable",
    "area_alternative",
    "lane_crosswalk",
    "lane_road_curb",
    "lane_single_white",
    "lane_double_white",
    "lane_single_yellow",
    "lane_double_yellow",
]

CLASS_ID_BY_NAME: Dict[str, int] = {name: idx for idx, name in enumerate(LANE_CLASS_NAMES)}


@dataclass(frozen=True)
class Config:
    data_root: Path
    output_root: Path
    splits: Tuple[str, ...]
    line_thickness: int
    min_contour_area: float
    max_points_per_contour: int
    max_files_per_split: Optional[int]
    link_mode: str
    resume: bool
    keep_empty_labels: bool
    image_width: int
    image_height: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Build a standalone YOLO-seg lane/drivable dataset from BDD100K lane JSON labels."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/bdd100k"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/traffic_sign_pipeline/iter_bdd100k_lanes"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="BDD100K splits to convert. test typically has no lane labels.",
    )
    parser.add_argument("--line-thickness", type=int, default=8)
    parser.add_argument("--min-contour-area", type=float, default=20.0)
    parser.add_argument("--max-points-per-contour", type=int, default=200)
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=None,
        help="Useful for pilot runs on very large datasets.",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy", "none"],
        help="How to place images in output/images. 'none' writes labels only.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip files with existing output label files.")
    parser.add_argument(
        "--keep-empty-labels",
        action="store_true",
        help="Write empty .txt labels for frames with no selected classes.",
    )
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)

    args = parser.parse_args()
    return Config(
        data_root=args.data_root,
        output_root=args.output_root,
        splits=tuple(args.splits),
        line_thickness=max(1, int(args.line_thickness)),
        min_contour_area=max(1.0, float(args.min_contour_area)),
        max_points_per_contour=max(8, int(args.max_points_per_contour)),
        max_files_per_split=args.max_files_per_split,
        link_mode=args.link_mode,
        resume=bool(args.resume),
        keep_empty_labels=bool(args.keep_empty_labels),
        image_width=max(32, int(args.image_width)),
        image_height=max(32, int(args.image_height)),
    )


def class_id_for_category(category: str) -> Optional[int]:
    category = str(category).strip().lower()

    mapping = {
        "area/drivable": "area_drivable",
        "area/alternative": "area_alternative",
        "lane/crosswalk": "lane_crosswalk",
        "lane/road curb": "lane_road_curb",
        "lane/single white": "lane_single_white",
        "lane/double white": "lane_double_white",
        "lane/single yellow": "lane_single_yellow",
        "lane/double yellow": "lane_double_yellow",
    }
    class_name = mapping.get(category)
    if class_name is not None:
        return CLASS_ID_BY_NAME[class_name]

    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def yolo_seg_line(class_id: int, contour: np.ndarray, width: int, height: int) -> str:
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / float(width), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(height), 0.0, 1.0)

    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
    return f"{class_id} {flat}"


def simplify_contour(contour: np.ndarray, max_points: int) -> np.ndarray:
    if len(contour) <= max_points:
        return contour

    eps = 0.001
    simplified = contour
    for _ in range(20):
        candidate = cv2.approxPolyDP(contour, eps, True)
        simplified = candidate
        if len(candidate) <= max_points:
            break
        eps *= 1.5
    return simplified


def draw_lane_object(mask: np.ndarray, points: np.ndarray, cls_id: int, thickness: int) -> None:
    if len(points) < 2:
        return

    if cls_id in {CLASS_ID_BY_NAME["area_drivable"], CLASS_ID_BY_NAME["area_alternative"]} and len(points) >= 3:
        cv2.fillPoly(mask, [points], color=255)
        return

    if cls_id == CLASS_ID_BY_NAME["lane_crosswalk"] and len(points) >= 3:
        area = cv2.contourArea(points.astype(np.float32))
        if abs(area) >= 1.0:
            cv2.fillPoly(mask, [points], color=255)
            return

    # Most lane categories are line-like annotations; rasterize with a fixed width.
    cv2.polylines(mask, [points], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)


def extract_labels_from_json(
    json_path: Path,
    width: int,
    height: int,
    line_thickness: int,
    min_contour_area: float,
    max_points_per_contour: int,
) -> List[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        return []

    objects = frames[0].get("objects", [])
    masks = {class_id: np.zeros((height, width), dtype=np.uint8) for class_id in range(len(LANE_CLASS_NAMES))}

    for obj in objects:
        cls_id = class_id_for_category(obj.get("category", ""))
        if cls_id is None:
            continue

        raw_poly = obj.get("poly2d")
        if not raw_poly or len(raw_poly) < 2:
            continue

        points: List[Tuple[int, int]] = []
        for entry in raw_poly:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            x = int(round(float(entry[0])))
            y = int(round(float(entry[1])))
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            points.append((x, y))

        if len(points) < 2:
            continue

        draw_lane_object(masks[cls_id], np.array(points, dtype=np.int32), cls_id, thickness=line_thickness)

    lines: List[str] = []
    for class_id, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            contour = simplify_contour(contour, max_points_per_contour)
            if len(contour) < 3:
                continue
            lines.append(yolo_seg_line(class_id, contour, width, height))

    return lines


def link_or_copy_image(src: Path, dst: Path, mode: str) -> None:
    if mode == "none":
        return

    if dst.exists():
        return

    ensure_dir(dst.parent)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def iter_label_files(split_labels_dir: Path, max_files: Optional[int]) -> Iterable[Path]:
    files = sorted(split_labels_dir.glob("*.json"))
    if max_files is not None:
        files = files[: max(0, max_files)]
    return files


def convert_split(cfg: Config, split: str) -> Dict[str, int]:
    labels_src_dir = cfg.data_root / "labels" / split
    images_src_dir = cfg.data_root / "bdd100k_images_100k" / "100k" / split

    labels_out_dir = cfg.output_root / "labels" / split
    images_out_dir = cfg.output_root / "images" / split

    ensure_dir(labels_out_dir)
    ensure_dir(images_out_dir)

    stats = {
        "json_total": 0,
        "converted": 0,
        "skipped_resume": 0,
        "missing_images": 0,
        "empty_labels": 0,
        "with_labels": 0,
    }

    for idx, json_path in enumerate(iter_label_files(labels_src_dir, cfg.max_files_per_split), start=1):
        stem = json_path.stem
        label_txt_path = labels_out_dir / f"{stem}.txt"
        image_src_path = images_src_dir / f"{stem}.jpg"
        image_out_path = images_out_dir / f"{stem}.jpg"

        stats["json_total"] += 1

        if cfg.resume and label_txt_path.exists() and (cfg.link_mode == "none" or image_out_path.exists()):
            stats["skipped_resume"] += 1
            continue

        if not image_src_path.exists():
            stats["missing_images"] += 1
            continue

        label_lines = extract_labels_from_json(
            json_path=json_path,
            width=cfg.image_width,
            height=cfg.image_height,
            line_thickness=cfg.line_thickness,
            min_contour_area=cfg.min_contour_area,
            max_points_per_contour=cfg.max_points_per_contour,
        )

        if label_lines:
            label_txt_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            stats["with_labels"] += 1
        elif cfg.keep_empty_labels:
            label_txt_path.write_text("", encoding="utf-8")
            stats["empty_labels"] += 1
        else:
            if label_txt_path.exists():
                label_txt_path.unlink()

        link_or_copy_image(image_src_path, image_out_path, cfg.link_mode)

        stats["converted"] += 1
        if idx % 500 == 0:
            print(
                f"[{split}] processed={idx} converted={stats['converted']} "
                f"with_labels={stats['with_labels']} empty={stats['empty_labels']}"
            )

    return stats


def write_dataset_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "dataset.yaml"
    payload = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(LANE_CLASS_NAMES),
        "names": LANE_CLASS_NAMES,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    cfg = parse_args()
    ensure_dir(cfg.output_root)

    print("Building BDD100K lane dataset (standalone formatter)...")
    print(f"data_root={cfg.data_root}")
    print(f"output_root={cfg.output_root}")
    print(f"splits={cfg.splits}")
    if cfg.max_files_per_split is not None:
        print(f"max_files_per_split={cfg.max_files_per_split}")

    all_stats: Dict[str, Dict[str, int]] = {}
    for split in cfg.splits:
        stats = convert_split(cfg, split)
        all_stats[split] = stats
        print(f"[{split}] {stats}")

    yaml_path = write_dataset_yaml(cfg.output_root)
    print(f"Dataset YAML written to: {yaml_path}")


if __name__ == "__main__":
    main()
