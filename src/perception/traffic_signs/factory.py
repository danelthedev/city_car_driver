import os
from typing import Type
from .base import BaseTrafficSignDetector
from .yolo import YOLODetector
from .faster_rcnn import FasterRCNNDetector
from .ssd import SSDDetector

class ModelFactory:
    """
    Factory to instantiate multiple kinds of object detection models dynamically.
    """
    
    _registry = {
        "yolo": YOLODetector,
        "fasterrcnn": FasterRCNNDetector,
        "ssd": SSDDetector,
    }

    @classmethod
    def create(cls, model_name: str, model_path: str = None, device: str = "cuda") -> BaseTrafficSignDetector:
        """
        Create a traffic sign detector instance.

        Args:
            model_name (str): Identifier for the model architecture (e.g., 'yolov8', 'fasterrcnn').
            model_path (str, optional): Path to saved model weights. Defaults to None.
            device (str, optional): Computation device ('cuda' or 'cpu'). Defaults to "cuda".

        Returns:
            BaseTrafficSignDetector: An instantiated detector.
        """
        model_name = model_name.lower().strip()
        if model_name not in cls._registry:
            raise ValueError(f"Unknown model architecture '{model_name}'. "
                             f"Supported architectures are: {list(cls._registry.keys())}")
        
        DetectorClass = cls._registry[model_name]
        return DetectorClass(model_path=model_path, device=device)
