from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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
