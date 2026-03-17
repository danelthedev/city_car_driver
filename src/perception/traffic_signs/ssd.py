from typing import Optional

from .base import BaseTrafficSignDetector

class SSDDetector(BaseTrafficSignDetector):
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path=model_path, device=device)
        self.load_model()
        
    def load_model(self):
        print(f"Loading SSD model on {self.device}...")
        pass

    def train(self, data_path: str, epochs: int, batch_size: int, **kwargs):
        print(f"Training SSD on {data_path} for {epochs} epochs...")
        pass
        
    def evaluate(self, data_path: str, **kwargs):
        print(f"Evaluating SSD on {data_path}...")
        return {"mAP": 0.0}

    def predict(self, image, confidence_threshold=0.5, **kwargs):
        # Stub for detection
        return []
