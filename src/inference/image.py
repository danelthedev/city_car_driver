import os

import cv2

from .detection import (
    run_detection_once,
    run_lane_segmentation_once,
    compute_intersection_status,
    resize_to_fit,
    resolve_class_name,
)
from .drawing import (
    draw_lane_instances,
    _draw_detection,
    _draw_legend,
    _tl_color,
    _SIGN_BOX_COLOR,
    _SIGN_TEXT_COLOR,
)


def run_inference_image(args, detector, tl_detector=None, lane_model=None):
    if not os.path.exists(args.image):
        print(f"Error: Target image file {args.image} does not exist.")
        return

    has_tl = tl_detector is not None
    has_lane = lane_model is not None
    tl_conf = args.tl_conf_threshold if args.tl_conf_threshold is not None else args.conf_threshold
    lane_conf = args.lane_conf_threshold if args.lane_conf_threshold is not None else args.conf_threshold
    lane_imgsz = args.lane_imgsz if args.lane_imgsz is not None else args.imgsz
    lane_scale = args.lane_inference_scale if args.lane_inference_scale is not None else args.inference_scale

    print(f"Running inference on image: {args.image}")
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image at {args.image}")
        return

    sign_detections, _ = run_detection_once(detector, frame, args.conf_threshold, args.imgsz, args.inference_scale)
    for (x1, y1, x2, y2, conf, cls) in sign_detections:
        class_name = resolve_class_name(detector, cls)
        _draw_detection(frame, x1, y1, x2, y2, f"{class_name}: {conf:.2f}", _SIGN_BOX_COLOR, _SIGN_TEXT_COLOR)

    if has_tl:
        tl_detections, _ = run_detection_once(tl_detector, frame, tl_conf, args.imgsz, args.inference_scale)
        for (x1, y1, x2, y2, conf, cls) in tl_detections:
            class_name = resolve_class_name(tl_detector, cls)
            color = _tl_color(class_name)
            _draw_detection(frame, x1, y1, x2, y2, f"TL-{class_name}: {conf:.2f}", color, color)

    if has_lane:
        lane_instances, _ = run_lane_segmentation_once(lane_model, frame, lane_conf, lane_imgsz, lane_scale)
        draw_lane_instances(frame, lane_instances, alpha=args.lane_mask_alpha)
        score, is_intersection, reasons = compute_intersection_status(
            lane_instances, frame.shape, threshold=args.intersection_threshold,
        )
        text = f"Intersection: {'YES' if is_intersection else 'no'} ({score:.2f})"
        if reasons:
            text += " | " + ",".join(reasons)
        cv2.putText(
            frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
            (20, 20, 220) if is_intersection else (80, 180, 80), 2,
        )

    _draw_legend(frame, has_tl, has_lane)

    window_name = f"{args.model} - Image Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)
    cv2.imshow(window_name, resize_to_fit(frame, args.window_width, args.window_height))
    print("Press any key inside the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
