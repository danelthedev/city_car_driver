from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2

from .taxonomy import BSTLD_CLASS_ID, BSTLD_CLASS_NAMES, BSTLD_LABEL_MAP
from .config import PipelineConfig
from .yolo_utils import (
    counter_to_dict,
    ensure_dataset_layout,
    format_yolo_line,
    prepare_dataset_dir,
    write_dataset_yaml,
    write_label_file,
    xyxy_to_yolo,
)


@dataclass
class BstldFrame:
    """One annotated frame from the Supervisely BSTLD export."""
    image_path: Path   # absolute path to the image file
    ann_path: Path     # absolute path to the companion JSON annotation


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
            candidates = [p for p in sorted(mapillary_img_dir.iterdir()) if p.is_file()]
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
                dst_lbl.write_text("", encoding="utf-8")
                n_negatives_added += 1

            print(
                f"[iter_bstld] Added {n_negatives_added} hard negative images from Mapillary "
                f"({len(empty_candidates)} empty candidates available)"
            )

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
