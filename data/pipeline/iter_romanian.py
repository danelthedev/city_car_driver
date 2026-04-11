from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .taxonomy import (
    BASE_CLASS_NAMES,
    ROMANIAN_DROP_CLASS_NAMES,
    ROMANIAN_FORCE_APPEND_CLASS_NAMES,
    ROMANIAN_TO_BASE_CLASS,
)
from .config import ClassManifest, PipelineConfig
from .yolo_utils import (
    counter_to_dict,
    parse_yolo_line_payload,
    prepare_dataset_dir,
    write_dataset_yaml,
    write_label_file,
)


def build_romanian_class_map(
    manifest: ClassManifest,
) -> Tuple[Dict[int, Optional[int]], List[Dict[str, object]]]:
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
                raise KeyError(
                    f"Forced Romanian class '{source_name}' is missing from unified class manifest."
                )
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
        raise FileNotFoundError(
            "Expected train and valid/val directories under Romanian source-root."
        )

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
