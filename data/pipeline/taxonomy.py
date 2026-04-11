from __future__ import annotations

from typing import Dict, List, Set, Tuple


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

SPEED_LIMIT_TO_CLASS_ID: Dict[int, int] = {
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
