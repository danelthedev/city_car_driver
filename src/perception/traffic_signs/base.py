from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseTrafficSignDetector(ABC):
    """
    Abstract base class for all traffic sign detection models.
    Enforces a standard API for loading models, training, evaluation, and prediction.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize the detector.

        Args:
            model_path (str, optional): Path to the model weights. Defaults to None.
            device (str, optional): Target device ("cuda" or "cpu"). Defaults to "cuda".
        """
        self.model_path = model_path
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model architecture and weights onto the specified device.
        """
        pass

    @abstractmethod
    def train(self, data_path: str, epochs: int, batch_size: int, **kwargs) -> Any:
        """
        Train the model on the specified dataset.

        Args:
            data_path (str): Path to the training dataset.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
        Returns:
            Any: Training history or metrics.
        """
        pass

    @abstractmethod
    def evaluate(self, data_path: str, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on a validation/test dataset.

        Args:
            data_path (str): Path to the evaluation dataset.
        Returns:
            Dict[str, float]: dictionary of evaluation metrics (e.g., mAP, precision, recall).
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        **kwargs,
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Detect traffic signs in an image.

        Args:
            image (np.ndarray): The input image in BGR format (OpenCV default).
            confidence_threshold (float, optional): Minimum confidence to return a bounding box.
            **kwargs: Optional backend-specific inference settings (e.g. image size).

        Returns:
            List[Tuple[int, int, int, int, float, int]]: A list of detections where each detection is
                (x_min, y_min, x_max, y_max, confidence, class_id).
        """
        pass
