from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .taxonomy import (
    BSTLD_CLASS_NAMES,
    DATASET_TARGET_ALIASES,
    DATASET_TARGETS,
    TARGET_DEPENDENCIES,
)
from .config import ClassManifest, PipelineConfig
from .yolo_utils import (
    existing_iterative_dataset_is_usable,
    validate_dataset,
    write_json,
)
from .manifest import build_class_manifest, class_manifest_to_dict
from .iter_mapillary import collect_train_backgrounds, prepare_iter_mapillary_backbone_dataset
from .iter_gtsdb import prepare_iter_gtsdb_plus_synthetic_dataset
from .iter_romanian import prepare_iter_romanian_dataset
from .iter_bstld import prepare_iter_bstld_dataset


def parse_requested_datasets(raw_value: str) -> List[str]:
    import re
    tokens = [token.strip().lower() for token in re.split(r"[\s,]+", raw_value) if token.strip()]
    if not tokens or "all" in tokens:
        return list(DATASET_TARGETS)

    canonical = [DATASET_TARGET_ALIASES.get(token, token) for token in tokens]
    invalid = sorted({token for token in canonical if token not in DATASET_TARGETS})
    if invalid:
        allowed = ", ".join(DATASET_TARGETS)
        raise ValueError(
            f"Unknown dataset target(s): {', '.join(invalid)}. Allowed values: all, {allowed}"
        )

    selected_set = set(canonical)
    return [target for target in DATASET_TARGETS if target in selected_set]


def expand_target_dependencies(
    requested: Sequence[str],
) -> Tuple[List[str], List[str]]:
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


def run_pipeline(cfg: PipelineConfig, requested_datasets: Sequence[str]) -> None:
    # Validate output root
    output_root = cfg.output_root.resolve()
    data_root = cfg.data_root.resolve()
    if output_root == data_root:
        raise ValueError("output_root must not be the same as data_root")
    output_root.mkdir(parents=True, exist_ok=True)

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
                    "No Mapillary train backgrounds available. "
                    "Generate iter_mapillary_backbone first."
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
        class_names = BSTLD_CLASS_NAMES if target == "iter_bstld" else manifest.names  # type: ignore[union-attr]
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
