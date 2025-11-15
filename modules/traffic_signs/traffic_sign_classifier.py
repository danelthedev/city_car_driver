from typing import List, Dict, Any
import numpy as np


class TrafficSignClassifier:

    def __init__(self, model_path: str = None, num_classes: int = 43):
        """
        Initialize the traffic sign classifier.

        Args:
            model_path: Path to the pre-trained model weights
            num_classes: Number of traffic sign classes to classify
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = None
        self.class_names = []
        self.input_shape = None

    def load_model(self, model_path: str = None) -> None:
        """
        Load the classification model from the specified path.

        Args:
            model_path: Path to the model weights file
        """
        raise NotImplementedError("Subclasses must implement load_model()")

    def train(self, train_data: Any, validation_data: Any = None,
              epochs: int = 10, batch_size: int = 32, **kwargs) -> Dict[str, Any]:
        """
        Train the classification model.

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

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify a single traffic sign image.

        Args:
            image: Input image as numpy array (BGR or RGB format)

        Returns:
            Dictionary containing:
                - 'class_id': Predicted class ID
                - 'class_name': Predicted class name
                - 'confidence': Prediction confidence score (0-1)
                - 'probabilities': Array of probabilities for all classes
        """
        raise NotImplementedError("Subclasses must implement classify()")

    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            save_path: Path where the model should be saved
        """
        raise NotImplementedError("Subclasses must implement save_model()")

    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary containing evaluation metrics (accuracy, loss, etc.)
        """
        raise NotImplementedError("Evaluate method should be implemented by subclasses")

    def get_class_names(self) -> List[str]:
        """
        Get the list of class names that the classifier can identify.

        Returns:
            List of class names
        """
        return self.class_names
