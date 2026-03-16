import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from ultralytics import YOLO

from .base import BaseTrafficSignDetector

class YOLODetector(BaseTrafficSignDetector):
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda"):
        # Allows for yolov8n, yolov8m, yolo11n, etc.
        super().__init__(model_path=model_path, device=device)
        self.load_model()
        
    def load_model(self):
        # We start with a pre-trained base model if the file doesn't exist or isn't provided
        weight_file = self.model_path if (self.model_path and os.path.exists(self.model_path)) else self.model_path or "yolov8n.pt"
        # The YOLO class from ultralytics uniquely handles v8, v9, v10, v11 seamlessly based on the string name
        self.model = YOLO(weight_file)
        self.model.to(self.device)

    def train(self, data_path: str, epochs: int = 100, batch_size: int = 16, **kwargs):
        # YOLOv8 expects a .yaml file configuring the dataset
        results = self.model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            device=self.device,
            **kwargs
        )
        return results

    def evaluate(self, data_path: str, **kwargs):
        results = self.model.val(data=data_path, device=self.device, **kwargs)
        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map
        }

    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int, float, int]]:
        results = self.model(image, conf=confidence_threshold, device=self.device, verbose=False)
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            detections.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
        
        return detections
