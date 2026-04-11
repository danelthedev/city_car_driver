import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Detection App")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test", "inference"],
        help="Mode to run the application in.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolo", "fasterrcnn", "ssd"],
        help="The detection model string identifier.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/traffic_sign_dataset/dataset.yaml",
        help="Path to the dataset config or root folder for train/test modes.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pre-trained model weights for the primary (sign) model.",
    )
    parser.add_argument(
        "--tl-weights",
        type=str,
        default=None,
        help="Path to a YOLO weights file for traffic-light detection.",
    )
    parser.add_argument(
        "--lane-weights",
        type=str,
        default=None,
        help="Path to a YOLO segmentation weights file for lane/drivable inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (cuda or cpu).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file for inference (instead of screen capture).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.20,
        help="Minimum confidence used during inference.",
    )
    parser.add_argument(
        "--tl-conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold for the traffic-light model. Defaults to --conf-threshold.",
    )
    parser.add_argument(
        "--lane-conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold for the lane model. Defaults to --conf-threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--inference-scale",
        type=float,
        default=1.0,
        help="Scale applied before inference only (0 < value <= 1).",
    )
    parser.add_argument(
        "--lane-imgsz",
        type=int,
        default=None,
        help="Inference image size for lane model. Defaults to --imgsz.",
    )
    parser.add_argument(
        "--lane-inference-scale",
        type=float,
        default=None,
        help="Scale applied before lane inference. Defaults to --inference-scale.",
    )
    parser.add_argument(
        "--sign-submit-interval-ms",
        type=float,
        default=0.0,
        help="Minimum time between sign async submissions in milliseconds (0 = unlimited).",
    )
    parser.add_argument(
        "--tl-submit-interval-ms",
        type=float,
        default=0.0,
        help="Minimum time between traffic-light async submissions in milliseconds (0 = unlimited).",
    )
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=1,
        help="Monitor index for screen capture (mss indexing: 1..N).",
    )
    parser.add_argument(
        "--async-capture",
        action="store_true",
        help="Capture frames in a background thread.",
    )
    parser.add_argument(
        "--max-draw-detections",
        type=int,
        default=40,
        help="Maximum detections to draw per frame (set <= 0 to draw all).",
    )
    parser.add_argument(
        "--min-confirm-frames",
        type=int,
        default=2,
        help="Minimum consecutive inference updates required before drawing a detection.",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=2,
        help="How many inference updates a confirmed detection can be missing before removal.",
    )
    parser.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.35,
        help="IoU threshold for matching detections across inference updates.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1400,
        help="Inference display window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Inference display window height in pixels.",
    )
    parser.add_argument(
        "--lane-mask-alpha",
        type=float,
        default=0.35,
        help="Transparency for lane segmentation overlays (0.0 to 1.0).",
    )
    parser.add_argument(
        "--intersection-threshold",
        type=float,
        default=0.55,
        help="Threshold for the intersection-ahead score from lane geometry cues.",
    )
    parser.add_argument(
        "--speed-telemetry",
        action="store_true",
        help="Enable speed extraction from process memory.",
    )

    return parser.parse_args()
