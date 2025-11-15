from typing import List, Dict, Any
import numpy as np


class TrafficSignDetector:

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the traffic sign detector.

        Args:
            model_path: Path to the pre-trained model weights
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []

    def load_model(self, model_path: str = None) -> None:
        """
        Load the detection model from the specified path.

        Args:
            model_path: Path to the model weights file
        """
        raise NotImplementedError("Subclasses must implement load_model()")

    def train(self, train_data: Any, validation_data: Any = None,
              epochs: int = 10, batch_size: int = 32, **kwargs) -> Dict[str, Any]:
        """
        Train the detection model.

        Args:
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training history and metrics
        """
        raise NotImplementedError("Subclasses must implement train()")

    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            save_path: Path where the model should be saved
        """
        raise NotImplementedError("Subclasses must implement save_model()")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect traffic signs in the given image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of detections, where each detection is a dictionary with:
                - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
                - 'confidence': Detection confidence score (0-1)
                - 'class_id': Class ID of the detected sign
                - 'class_name': Class name of the detected sign
        """
        raise NotImplementedError("Subclasses must implement detect()")

    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary containing evaluation metrics (mAP, precision, recall, etc.)
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def get_class_names(self) -> List[str]:
        """
        Get the list of class names that the detector can identify.

        Returns:
            List of class names
        """
        return self.class_names
