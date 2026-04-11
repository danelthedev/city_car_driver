from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .taxonomy import (
    BASE_CLASS_NAMES,
    ROMANIAN_DROP_CLASS_NAMES,
    ROMANIAN_FORCE_APPEND_CLASS_NAMES,
    ROMANIAN_TO_BASE_CLASS,
)
from .config import ClassManifest, PipelineConfig
from .yolo_utils import read_yaml


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
        raise ValueError(
            f"Romanian data.yaml nc ({declared_nc}) does not match names length ({len(names)})."
        )
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
