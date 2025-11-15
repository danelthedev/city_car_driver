from modules.traffic_signs.traffic_sign_detector import TrafficSignDetector
from typing import List, Dict, Any
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch


class YoloTrafficSignDetector(TrafficSignDetector):
    """YOLO-based traffic sign detector using Ultralytics YOLO."""

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25,
                 img_size: int = 416, use_half: bool = True, save_crops: bool = False,
                 multi_scale: bool = False, scales: tuple = (1.0,), tta: bool = False,
                 adaptive_conf: bool = False, min_conf_small: float = 0.10, small_area_ratio: float = 0.002,
                 agnostic_nms: bool = False, max_det: int = 300):
        """
        Initialize the YOLO traffic sign detector.

        Args:
            model_path: Path to the pre-trained YOLO model weights
            confidence_threshold: Minimum confidence score for detections (lower = more detections)
            img_size: Input image size for inference (smaller = faster, 416 recommended for speed)
            use_half: Use FP16 half-precision for faster inference (GPU only)
            save_crops: Whether to save cropped detections to tmp folder
            multi_scale: Enable multi-scale inference
            scales: Tuple of scales for multi-scale inference
            tta: Enable test-time augmentation
            adaptive_conf: Enable adaptive confidence thresholding
            min_conf_small: Minimum confidence for small objects (adaptive)
            small_area_ratio: Area ratio threshold for small objects (adaptive)
            agnostic_nms: Use class-agnostic NMS
            max_det: Maximum number of detections per image
        """
        super().__init__(model_path, confidence_threshold)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = img_size
        self.use_half = use_half and self.device == 'cuda'  # Only use half precision on GPU
        self.save_crops = save_crops
        # New inference controls
        self.multi_scale = multi_scale
        self.scales = scales if multi_scale else (1.0,)
        self.tta = tta
        self.adaptive_conf = adaptive_conf
        self.min_conf_small = min_conf_small
        self.small_area_ratio = small_area_ratio
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        print(f"Using device: {self.device}")
        print(f"Inference settings: base_img_size={img_size}, half_precision={self.use_half}, conf_threshold={confidence_threshold}")
        if self.multi_scale:
            print(f"Multi-scale enabled with scales={self.scales}")
        if self.adaptive_conf:
            print(f"Adaptive confidence enabled (min_conf_small={self.min_conf_small}, small_area_ratio={self.small_area_ratio})")

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Use nano model for faster inference
            self.model = YOLO('yolo11s.pt')
            print("Initialized with YOLO11s base model (optimized for speed)")

    def load_model(self, model_path: str = None) -> None:
        """
        Load the YOLO model from the specified path.

        Args:
            model_path: Path to the model weights file
        """
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            raise ValueError("No model path specified")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.model.to(self.device)

        # Note: Half precision is applied during inference only, not during model loading
        # This prevents dtype conflicts during training and evaluation

        self.model_path = str(model_path)

        # Extract class names from model
        if hasattr(self.model, 'names'):
            self.class_names = list(self.model.names.values())

        print(f"Model loaded from {model_path}")
        print(f"Number of classes: {len(self.class_names)}")

    def train(self, train_data: Any, validation_data: Any = None,
              epochs: int = 10, batch_size: int = 32, **kwargs) -> Dict[str, Any]:
        """
        Train the YOLO detection model.

        Args:
            train_data: Path to YAML configuration file or dataset path
            validation_data: Validation dataset (included in YAML config)
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional training parameters (imgsz, patience, etc.)

        Returns:
            Dictionary containing training history and metrics
        """
        if self.model is None:
            self.model = YOLO('yolo11s.pt')

        # Default training parameters
        train_params = {
            'data': train_data,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': kwargs.get('imgsz', 640),
            'device': self.device,
            'patience': kwargs.get('patience', 50),
            'save': kwargs.get('save', True),
            'project': kwargs.get('project', 'models'),
            'name': kwargs.get('name', 'traffic_sign_detector'),
            'exist_ok': kwargs.get('exist_ok', True),
            'pretrained': kwargs.get('pretrained', True),
            'optimizer': kwargs.get('optimizer', 'auto'),
            'verbose': kwargs.get('verbose', True),
            'seed': kwargs.get('seed', 0),
            'deterministic': kwargs.get('deterministic', True),
            'single_cls': kwargs.get('single_cls', False),
            'rect': kwargs.get('rect', False),
            'cos_lr': kwargs.get('cos_lr', False),
            'close_mosaic': kwargs.get('close_mosaic', 10),
            'resume': kwargs.get('resume', False),
            'amp': kwargs.get('amp', True),
            'fraction': kwargs.get('fraction', 1.0),
            'profile': kwargs.get('profile', False),
            'lr0': kwargs.get('lr0', 0.01),
            'lrf': kwargs.get('lrf', 0.01),
            'momentum': kwargs.get('momentum', 0.937),
            'weight_decay': kwargs.get('weight_decay', 0.0005),
            'warmup_epochs': kwargs.get('warmup_epochs', 3.0),
            'warmup_momentum': kwargs.get('warmup_momentum', 0.8),
            'warmup_bias_lr': kwargs.get('warmup_bias_lr', 0.1),
            'box': kwargs.get('box', 7.5),
            'cls': kwargs.get('cls', 0.5),
            'dfl': kwargs.get('dfl', 1.5),
            'plots': kwargs.get('plots', True),
            'val': kwargs.get('val', True),
        }

        print(f"Starting training on {self.device}...")
        print(f"Training parameters: epochs={epochs}, batch_size={batch_size}")

        # Train the model
        results = self.model.train(**train_params)

        # Extract class names after training
        if hasattr(self.model, 'names'):
            self.class_names = list(self.model.names.values())

        # Prepare training history
        history = {
            'results': results,
            'model_path': self.model.trainer.last if hasattr(self.model, 'trainer') else None,
            'best_model_path': self.model.trainer.best if hasattr(self.model, 'trainer') else None,
        }

        print("Training completed!")
        if history['best_model_path']:
            print(f"Best model saved to: {history['best_model_path']}")

        return history

    def save_model(self, save_path: str) -> None:
        """
        Save the trained YOLO model to disk.

        Args:
            save_path: Path where the model should be saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Export the model
        self.model.save(str(save_path))
        self.model_path = str(save_path)

        print(f"Model saved to {save_path}")

    @staticmethod
    def _iou(boxA, boxB):
        # Basic IoU for NMS merging
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interW = max(0.0, xB - xA); interH = max(0.0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0.0, boxA[2]-boxA[0]) * max(0.0, boxA[3]-boxA[1])
        boxBArea = max(0.0, boxB[2]-boxB[0]) * max(0.0, boxB[3]-boxB[1])
        union = boxAArea + boxBArea - interArea + 1e-9
        return interArea / union

    @staticmethod
    def _nms(detections, iou_thresh=0.55, agnostic=False):
        if not detections: return []
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            remaining = []
            for det in detections:
                if not agnostic and det['class_id'] != best['class_id']:
                    remaining.append(det); continue
                if YoloTrafficSignDetector._iou(best['bbox'], det['bbox']) < iou_thresh:
                    remaining.append(det)
            detections = remaining
        return kept

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect traffic signs in the given image using YOLO.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of detections, where each detection is a dictionary with:
                - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
                - 'confidence': Detection confidence score (0-1)
                - 'class_id': Class ID of the detected sign
                - 'class_name': Class name of the detected sign
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        h, w = image.shape[:2]
        img_area = float(h * w)
        collected = []
        for scale in self.scales:
            scaled_size = int(self.img_size * scale)
            # Use adaptive confidence to set YOLO's internal threshold
            inference_conf = self.confidence_threshold
            if self.adaptive_conf:
                # For adaptive mode, use the minimum threshold and filter later
                inference_conf = self.min_conf_small

            results = self.model(
                image,
                conf=inference_conf,
                iou=0.45,
                imgsz=scaled_size,
                half=self.use_half,
                device=self.device,
                verbose=False,
                agnostic_nms=self.agnostic_nms,
                max_det=self.max_det,
                augment=self.tta
            )
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else str(cls_id)
                    bw = max(0.0, xyxy[2]-xyxy[0]); bh = max(0.0, xyxy[3]-xyxy[1])
                    b_area = bw * bh
                    area_ratio = b_area / (img_area + 1e-9)

                    # Apply adaptive confidence thresholding
                    if self.adaptive_conf:
                        # Small objects: use lower threshold
                        if area_ratio < self.small_area_ratio:
                            if conf < self.min_conf_small:
                                continue
                        # Larger objects: use normal threshold
                        else:
                            if conf < self.confidence_threshold:
                                continue
                    # Non-adaptive: simple threshold check (already done by YOLO, but double-check)
                    elif conf < self.confidence_threshold:
                        continue

                    collected.append({
                        'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'scale': scale,
                        'area_ratio': area_ratio
                    })
        if self.multi_scale and len(collected) > 1:
            detections = self._nms(collected, iou_thresh=0.55, agnostic=self.agnostic_nms)
        else:
            detections = collected
        if self.save_crops and detections:
            tmp_dir = Path("tmp"); tmp_dir.mkdir(exist_ok=True)
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = [int(c) for c in det['bbox']]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                crop = image[y1:y2, x1:x2]
                cv2.imwrite(str(tmp_dir / f"sign_{i}.png"), crop)
        return detections

    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the YOLO model on test data.

        Args:
            test_data: Path to test dataset YAML file or validation split

        Returns:
            Dictionary containing evaluation metrics (mAP, precision, recall, etc.)
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")

        print(f"Evaluating model on {self.device}...")

        # Run validation (always use float32 for evaluation to avoid dtype issues)
        results = self.model.val(data=test_data, device=self.device, half=False)

        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50) if hasattr(results, 'box') else 0.0,
            'mAP50-95': float(results.box.map) if hasattr(results, 'box') else 0.0,
            'precision': float(results.box.mp) if hasattr(results, 'box') else 0.0,
            'recall': float(results.box.mr) if hasattr(results, 'box') else 0.0,
        }

        print(f"Evaluation metrics: {metrics}")

        return metrics

    def get_class_names(self) -> List[str]:
        """
        Get the list of class names that the detector can identify.

        Returns:
            List of class names
        """
        if self.model is not None and hasattr(self.model, 'names'):
            return list(self.model.names.values())
        return self.class_names

    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]] = None,
                           save_path: str = None, show: bool = False) -> np.ndarray:
        """
        Visualize detections on the image.

        Args:
            image: Input image as numpy array (BGR format)
            detections: List of detections (if None, will run detection)
            save_path: Path to save the annotated image
            show: Whether to display the image

        Returns:
            Annotated image
        """
        if detections is None:
            detections = self.detect(image)

        # Create a copy of the image for annotation
        annotated_image = image.copy()

        # Draw each detection
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            # Get label size for background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw label background
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness
            )

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated_image)
            print(f"Annotated image saved to {save_path}")

        # Show if requested
        if show:
            cv2.imshow("Traffic Sign Detections", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_image

