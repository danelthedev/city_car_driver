from modules.traffic_signs.traffic_sign_detector import TrafficSignDetector
from typing import List, Dict, Any
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch


class YoloTrafficSignDetector(TrafficSignDetector):
    """
    YOLO-based traffic sign detector for detecting all traffic signs (single class).
    Uses YOLO11n or YOLO11s for fast and accurate detection.
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25,
                 model_size: str = 'n'):
        """
        Initialize the YOLO traffic sign detector.

        Args:
            model_path: Path to a trained model. If None, loads a pretrained YOLO model.
            confidence_threshold: Minimum confidence score for detections (0-1)
            model_size: YOLO model size ('n' for nano, 's' for small)
        """
        super().__init__(model_path, confidence_threshold)
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Initialize with pretrained YOLO model
            self.model = YOLO(f'yolo11{model_size}.pt')

        self.class_names = ['traffic_sign']

    def load_model(self, model_path: str = None) -> None:
        """
        Load the detection model from the specified path.

        Args:
            model_path: Path to the model weights file
        """
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            raise ValueError("No model path provided")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(str(model_path))
        self.model_path = str(model_path)

        # Update class names from model if available
        if hasattr(self.model, 'names'):
            self.class_names = list(self.model.names.values())

    def train(self, data_yaml: str, epochs: int = 100, batch_size: int = 16,
              imgsz: int = 640, project: str = 'models', name: str = 'yolo_detector',
              **kwargs) -> Dict[str, Any]:
        """
        Train the YOLO detection model.

        Args:
            data_yaml: Path to the data.yaml file containing dataset configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
            project: Project directory to save results
            name: Experiment name
            **kwargs: Additional training parameters for YOLO

        Returns:
            Dictionary containing training results and metrics
        """
        if self.model is None:
            self.model = YOLO(f'yolo11{self.model_size}.pt')

        print(f"Starting YOLO training with {self.model_size} model...")
        print(f"Device: {self.device}")
        print(f"Dataset: {data_yaml}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")

        # Train the model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            device=self.device,
            # patience=20,  # Early stopping patience
            # save=True,
            # save_period=10,  # Save checkpoint every 10 epochs
            # plots=True,  # Generate plots
            **kwargs
        )

        # Update model path to best weights
        self.model_path = str(Path(project) / name / 'weights' / 'best.pt')

        # Load the best model
        self.model = YOLO(self.model_path)

        print(f"\nTraining completed! Best model saved to: {self.model_path}")

        return {
            'results': results,
            'best_model_path': self.model_path
        }

    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            save_path: Path where the model should be saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # YOLO models save automatically during training
        # This method copies the current model to a new location
        import shutil
        if self.model_path and Path(self.model_path).exists():
            shutil.copy(self.model_path, save_path)
            print(f"Model saved to: {save_path}")
        else:
            print("Warning: No trained model path found. Model may not be properly saved.")

    def detect(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Detect traffic signs in the given image.

        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Override default confidence threshold

        Returns:
            List of detections, where each detection is a dictionary with:
                - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
                - 'confidence': Detection confidence score (0-1)
                - 'class_id': Class ID (always 0 for single-class)
                - 'class_name': Class name ('traffic_sign')
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")

        if conf_threshold is None:
            conf_threshold = self.confidence_threshold

        # Run inference
        results = self.model.predict(
            image,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
            augment=True
        )

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get bounding box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())

                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                }
                detections.append(detection)

        return detections

    def evaluate(self, data_yaml: str = None, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on validation/test data.

        Args:
            data_yaml: Path to data.yaml file (uses training data if None)
            **kwargs: Additional validation parameters

        Returns:
            Dictionary containing evaluation metrics (mAP, precision, recall, etc.)
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")

        print("Evaluating model...")

        # Run validation
        metrics = self.model.val(
            data=data_yaml,
            device=self.device,
            **kwargs
        )

        results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }

        print(f"\nEvaluation Results:")
        print(f"  mAP@50: {results['mAP50']:.4f}")
        print(f"  mAP@50-95: {results['mAP50-95']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")

        return results

    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]],
                            save_path: str = None, show: bool = False) -> np.ndarray:
        """
        Visualize detections on the image.

        Args:
            image: Input image
            detections: List of detections from detect()
            save_path: Path to save the annotated image
            show: Whether to display the image

        Returns:
            Annotated image
        """
        annotated_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw label background
            cv2.rectangle(annotated_image,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         (0, 255, 0), -1)

            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated_image)
            print(f"Annotated image saved to: {save_path}")

        if show:
            cv2.imshow('Detections', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_image

    def get_class_names(self) -> List[str]:
        """
        Get the list of class names that the detector can identify.

        Returns:
            List of class names
        """
        return self.class_names
