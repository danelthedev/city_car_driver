import time

import cv2
import numpy as np


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def resolve_class_name(detector, class_id: int) -> str:
    model = getattr(detector, "model", None)
    names = getattr(model, "names", None) if model is not None else None

    if isinstance(names, dict):
        if class_id in names:
            return str(names[class_id])
        str_id = str(class_id)
        if str_id in names:
            return str(names[str_id])
    elif isinstance(names, (list, tuple)):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    return f"class-{class_id}"


def resize_to_fit(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    image_h, image_w = image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return image

    scale = min(float(target_width) / float(image_w), float(target_height) / float(image_h))
    scale = max(scale, 0.01)
    resized_w = max(1, int(round(image_w * scale)))
    resized_h = max(1, int(round(image_h * scale)))

    if resized_w == image_w and resized_h == image_h:
        return image
    return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)


def scale_detections_to_original(detections, scale: float, frame_width: int, frame_height: int):
    if not detections or scale == 1.0:
        return detections

    inv_scale = 1.0 / scale
    max_x = max(frame_width - 1, 0)
    max_y = max(frame_height - 1, 0)
    scaled = []

    for (x1, y1, x2, y2, conf, cls) in detections:
        sx1 = max(0, min(max_x, int(round(x1 * inv_scale))))
        sy1 = max(0, min(max_y, int(round(y1 * inv_scale))))
        sx2 = max(0, min(max_x, int(round(x2 * inv_scale))))
        sy2 = max(0, min(max_y, int(round(y2 * inv_scale))))

        if sx2 <= sx1 or sy2 <= sy1:
            continue
        scaled.append((sx1, sy1, sx2, sy2, conf, cls))

    return scaled


def scale_bbox_to_frame(
    x1: int, y1: int, x2: int, y2: int,
    src_width: int, src_height: int,
    dst_width: int, dst_height: int,
):
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return x1, y1, x2, y2

    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)

    tx1 = max(0, min(dst_width - 1, int(round(x1 * sx))))
    ty1 = max(0, min(dst_height - 1, int(round(y1 * sy))))
    tx2 = max(0, min(dst_width - 1, int(round(x2 * sx))))
    ty2 = max(0, min(dst_height - 1, int(round(y2 * sy))))

    if tx2 <= tx1:
        tx2 = min(dst_width - 1, tx1 + 1)
    if ty2 <= ty1:
        ty2 = min(dst_height - 1, ty1 + 1)

    return tx1, ty1, tx2, ty2


def run_detection_once(
    detector,
    frame: np.ndarray,
    confidence_threshold: float,
    image_size: int,
    inference_scale: float,
):
    start_time = time.perf_counter()
    frame_h, frame_w = frame.shape[:2]

    scale = float(inference_scale)
    if scale <= 0.0:
        scale = 1.0

    inference_frame = frame
    if scale != 1.0:
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        inference_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

    detections = detector.predict(
        inference_frame,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
    )
    detections = scale_detections_to_original(detections, scale, frame_w, frame_h)
    duration = max(time.perf_counter() - start_time, 1e-6)
    return detections, duration


def run_lane_segmentation_once(
    lane_model,
    frame: np.ndarray,
    confidence_threshold: float,
    image_size: int,
    inference_scale: float,
):
    start_time = time.perf_counter()
    frame_h, frame_w = frame.shape[:2]

    scale = float(inference_scale)
    if scale <= 0.0:
        scale = 1.0

    inference_frame = frame
    if scale != 1.0:
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        inference_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

    results = lane_model(
        inference_frame,
        conf=confidence_threshold,
        imgsz=max(320, int(image_size)),
        verbose=False,
    )
    result = results[0]

    lane_instances = []
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)

    if masks is not None and boxes is not None and len(masks.data) > 0 and len(boxes) > 0:
        mask_data = masks.data.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(mask_data),), dtype=np.float32)
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(mask_data),), dtype=np.float32)

        for idx in range(min(len(mask_data), len(confs), len(classes))):
            raw_mask = (mask_data[idx] > 0.5).astype(np.uint8)
            if raw_mask.sum() <= 0:
                continue

            if raw_mask.shape[1] != frame_w or raw_mask.shape[0] != frame_h:
                raw_mask = cv2.resize(raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

            class_id = int(classes[idx])
            if isinstance(result.names, dict):
                class_name = str(result.names.get(class_id, f"class-{class_id}"))
            elif isinstance(result.names, (list, tuple)) and 0 <= class_id < len(result.names):
                class_name = str(result.names[class_id])
            else:
                class_name = f"class-{class_id}"

            lane_instances.append({
                "mask": raw_mask,
                "class_id": class_id,
                "class_name": class_name,
                "conf": float(confs[idx]),
            })

    duration = max(time.perf_counter() - start_time, 1e-6)
    return lane_instances, duration


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area) / float(union)


def apply_temporal_consensus(
    detections,
    tracks: dict,
    next_track_id: int,
    min_confirm_frames: int,
    max_missing_frames: int,
    match_iou_threshold: float,
):
    sorted_detections = sorted(detections, key=lambda item: item[4], reverse=True)
    matched_track_ids = set()

    for (x1, y1, x2, y2, conf, cls) in sorted_detections:
        best_track_id = None
        best_iou = match_iou_threshold

        for track_id, track in tracks.items():
            if track_id in matched_track_ids:
                continue
            if int(track["cls"]) != int(cls):
                continue
            iou = compute_iou((x1, y1, x2, y2), track["bbox"])
            if iou >= best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            tracks[next_track_id] = {
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(conf),
                "cls": int(cls),
                "hits": 1,
                "missed": 0,
            }
            matched_track_ids.add(next_track_id)
            next_track_id += 1
        else:
            track = tracks[best_track_id]
            track["bbox"] = (int(x1), int(y1), int(x2), int(y2))
            track["conf"] = float(conf)
            track["hits"] = int(track["hits"]) + 1
            track["missed"] = 0
            matched_track_ids.add(best_track_id)

    stale_track_ids = []
    stable_detections = []

    for track_id, track in tracks.items():
        if track_id not in matched_track_ids:
            track["missed"] = int(track["missed"]) + 1

        if int(track["missed"]) > max_missing_frames:
            stale_track_ids.append(track_id)
            continue

        if int(track["hits"]) >= min_confirm_frames:
            tx1, ty1, tx2, ty2 = track["bbox"]
            stable_detections.append((tx1, ty1, tx2, ty2, float(track["conf"]), int(track["cls"])))

    for track_id in stale_track_ids:
        del tracks[track_id]

    stable_detections.sort(key=lambda item: item[4], reverse=True)
    return stable_detections, next_track_id


def compute_intersection_status(lane_instances, frame_shape, threshold: float = 0.55):
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, False, []

    x1 = int(0.20 * w)
    x2 = int(0.80 * w)
    y1 = int(0.35 * h)
    y2 = int(0.90 * h)
    roi_area = float(max(x2 - x1, 1) * max(y2 - y1, 1))

    drivable_mask = np.zeros((h, w), dtype=np.uint8)
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    crosswalk_mask = np.zeros((h, w), dtype=np.uint8)

    for inst in lane_instances:
        mask = inst.get("mask")
        if mask is None:
            continue
        cls = _normalize_name(inst.get("class_name", ""))
        if cls in {"area_drivable", "area_alternative"}:
            drivable_mask = np.maximum(drivable_mask, mask)
        elif cls == "lane_crosswalk":
            crosswalk_mask = np.maximum(crosswalk_mask, mask)
        elif cls.startswith("lane_"):
            lane_mask = np.maximum(lane_mask, mask)

    drivable_ratio = float(np.count_nonzero(drivable_mask[y1:y2, x1:x2])) / roi_area
    lane_ratio = float(np.count_nonzero(lane_mask[y1:y2, x1:x2])) / roi_area
    crosswalk_ratio = float(np.count_nonzero(crosswalk_mask[y1:y2, x1:x2])) / roi_area

    upper = lane_mask[int(0.35 * h):int(0.60 * h), x1:x2]
    num_components, _ = cv2.connectedComponents((upper > 0).astype(np.uint8))
    branching_components = max(0, int(num_components) - 1)

    score = 0.0
    reasons = []

    if crosswalk_ratio > 0.012:
        score += 0.45
        reasons.append("crosswalk")
    if drivable_ratio > 0.20:
        score += 0.25
        reasons.append("wide-drivable")
    if lane_ratio > 0.02 and branching_components >= 3:
        score += 0.30
        reasons.append("lane-branching")

    score = max(0.0, min(1.0, score))
    return score, score >= float(threshold), reasons
