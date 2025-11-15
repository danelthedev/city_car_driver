from modules.traffic_signs.traffic_sign_detector import TrafficSignDetector
from typing import List, Dict, Any
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch


class YoloTrafficSignDetector(TrafficSignDetector):
    """YOLO-based traffic sign detector using Ultralytics YOLO."""

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the YOLO traffic sign detector.

        Args:
            model_path: Path to the pre-trained YOLO model weights
            confidence_threshold: Minimum confidence score for detections
        """
        super().__init__(model_path, confidence_threshold)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Initialize with YOLOv8 nano model for traffic sign detection
            self.model = YOLO('yolov8n.pt')
            print("Initialized with YOLOv8n base model")

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
            self.model = YOLO('yolov8n.pt')

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

        # Run inference
        results = self.model(image, conf=self.confidence_threshold, device=self.device, verbose=False)

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy()

                # Extract confidence
                confidence = float(box.conf[0].cpu().numpy())

                # Extract class
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id] if hasattr(self.model, 'names') else str(class_id)

                detection = {
                    'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }

                detections.append(detection)

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

        # Run validation
        results = self.model.val(data=test_data, device=self.device)

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
