from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml


# Must match the canonical 43-class ordering used by the existing pipeline.
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


# Clear semantic matches only (conservative mapping policy).
ROMANIAN_TO_EXISTING_CLASS: Dict[str, str] = {
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


DROP_CLASS_NAMES = {
    "forb_speed_over_5",
    "forb_speed_over_10",
    "forb_speed_over_40",
    "forb_speed_over_90",
    "forb_speed_over_130",
}


def read_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yolo_labels(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def parse_yolo_line(line: str) -> Optional[Tuple[int, str]]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        source_id = int(parts[0])
    except ValueError:
        return None
    return source_id, " ".join(parts[1:])


def find_split_root(source_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        split_root = source_root / candidate
        if split_root.exists() and split_root.is_dir():
            return split_root
    return None


def build_class_mapping(romanian_names: List[str]) -> Tuple[Dict[int, Optional[int]], List[str], List[Dict[str, object]]]:
    base_id_by_name = {name: idx for idx, name in enumerate(BASE_CLASS_NAMES)}
    next_id = len(BASE_CLASS_NAMES)
    class_map: Dict[int, Optional[int]] = {}
    extended_names = list(BASE_CLASS_NAMES)
    map_table: List[Dict[str, object]] = []

    for source_id, source_name in enumerate(romanian_names):
        if source_name in DROP_CLASS_NAMES:
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

        target_existing_name = ROMANIAN_TO_EXISTING_CLASS.get(source_name)
        if target_existing_name is not None:
            target_id = base_id_by_name[target_existing_name]
            class_map[source_id] = target_id
            map_table.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "action": "map-existing",
                    "target_id": target_id,
                    "target_name": target_existing_name,
                    "reason": "conservative-semantic-match",
                }
            )
            continue

        class_map[source_id] = next_id
        extended_names.append(source_name)
        map_table.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "action": "append-new",
                "target_id": next_id,
                "target_name": source_name,
                "reason": "no-clean-match-in-base-43",
            }
        )
        next_id += 1

    return class_map, extended_names, map_table


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def convert_split(
    split_name: str,
    source_split_root: Path,
    output_root: Path,
    class_map: Dict[int, Optional[int]],
    source_names: List[str],
    source_counts: Dict[str, Counter],
    target_counts: Dict[str, Counter],
    dropped_by_source_name: Counter,
) -> Dict[str, int]:
    source_images = source_split_root / "images"
    source_labels = source_split_root / "labels"
    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"Missing images/labels folders under: {source_split_root}")

    out_images = output_root / "images" / split_name
    out_labels = output_root / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted([path for path in source_images.iterdir() if path.is_file()])
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
            write_yolo_labels(destination_label, [])
            continue

        converted_lines: List[str] = []
        for raw_line in source_label.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue

            parsed = parse_yolo_line(raw_line)
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

        write_yolo_labels(destination_label, converted_lines)

    return stats


def write_dataset_yaml(output_root: Path, names: List[str], include_test: bool) -> Path:
    yaml_data: Dict[str, object] = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    if include_test:
        yaml_data["test"] = "images/test"

    yaml_path = output_root / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(yaml_data, stream, sort_keys=False)
    return yaml_path


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Convert Romanian YOLO dataset labels into the existing traffic-sign taxonomy: "
            "preserve base 43 classes, append Romanian-only classes, and drop unsupported speed classes."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_root / "romanian_traffic_signs",
        help="Source Romanian YOLO dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_root / "traffic_sign_pipeline" / "romanian_finetune",
        help="Output dataset root",
    )
    parser.add_argument(
        "--no-overwrite-output",
        action="store_true",
        help="Fail if output-root exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root: Path = args.source_root
    output_root: Path = args.output_root

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")

    config_path = source_root / "data.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing source config: {config_path}")

    config = read_yaml(config_path)
    source_names: List[str] = list(config.get("names", []))
    declared_nc = int(config.get("nc", len(source_names)))
    if declared_nc != len(source_names):
        raise ValueError(
            f"data.yaml nc ({declared_nc}) does not match names length ({len(source_names)})."
        )

    class_map, extended_names, map_table = build_class_mapping(source_names)

    train_root = find_split_root(source_root, ("train",))
    val_root = find_split_root(source_root, ("valid", "val"))
    test_root = find_split_root(source_root, ("test",))
    if train_root is None or val_root is None:
        raise FileNotFoundError("Expected train and valid/val directories under source-root.")

    prepare_output_root(output_root, overwrite=not args.no_overwrite_output)

    source_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    target_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    dropped_by_source_name: Counter = Counter()

    print("[1/4] Converting train split...")
    train_stats = convert_split(
        split_name="train",
        source_split_root=train_root,
        output_root=output_root,
        class_map=class_map,
        source_names=source_names,
        source_counts=source_counts,
        target_counts=target_counts,
        dropped_by_source_name=dropped_by_source_name,
    )

    print("[2/4] Converting val split...")
    val_stats = convert_split(
        split_name="val",
        source_split_root=val_root,
        output_root=output_root,
        class_map=class_map,
        source_names=source_names,
        source_counts=source_counts,
        target_counts=target_counts,
        dropped_by_source_name=dropped_by_source_name,
    )

    if test_root is not None and (test_root / "images").exists() and (test_root / "labels").exists():
        source_counts["test"] = Counter()
        target_counts["test"] = Counter()
        print("[3/4] Converting test split...")
        test_stats = convert_split(
            split_name="test",
            source_split_root=test_root,
            output_root=output_root,
            class_map=class_map,
            source_names=source_names,
            source_counts=source_counts,
            target_counts=target_counts,
            dropped_by_source_name=dropped_by_source_name,
        )
        include_test = True
    else:
        test_stats = None
        include_test = False

    print("[4/4] Writing YAML and conversion reports...")
    yaml_path = write_dataset_yaml(output_root, extended_names, include_test=include_test)

    reports_dir = output_root / "reports"
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
        if row["action"] == "append-new"
    ]

    summary = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "base_nc": len(BASE_CLASS_NAMES),
        "source_nc": len(source_names),
        "target_nc": len(extended_names),
        "source_names": source_names,
        "target_names": extended_names,
        "mapping_policy": {
            "unsupported_speed_classes_dropped": sorted(DROP_CLASS_NAMES),
            "conservative_existing_map_count": len(ROMANIAN_TO_EXISTING_CLASS),
            "append_new_classes": True,
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

    summary_path = reports_dir / "remap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print("Dataset YAML:", yaml_path)
    print("Class mapping:", map_json_path)
    print("Summary report:", summary_path)


if __name__ == "__main__":
    main()
