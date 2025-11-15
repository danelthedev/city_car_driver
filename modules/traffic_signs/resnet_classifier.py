from modules.traffic_signs.traffic_sign_classifier import TrafficSignClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any, List
from tqdm import tqdm
import time


class GTSRBDataset(Dataset):
    """GTSRB Dataset loader."""

    def __init__(self, root_dir: str, transform=None, is_test=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_test = is_test
        self.images = []
        self.labels = []

        if not is_test:
            # Load training data
            images_dir = self.root_dir / 'GTSRB' / 'Final_Training' / 'Images'
            for class_dir in sorted(images_dir.iterdir()):
                if class_dir.is_dir():
                    class_id = int(class_dir.name)
                    for img_path in class_dir.glob('*.ppm'):
                        self.images.append(str(img_path))
                        self.labels.append(class_id)
        else:
            # Load test data
            import pandas as pd
            test_csv = self.root_dir / 'GT-final_test.csv'
            test_dir = self.root_dir / 'GTSRB' / 'Final_Test' / 'Images'

            if test_csv.exists():
                df = pd.read_csv(test_csv, sep=';')
                for _, row in df.iterrows():
                    img_path = test_dir / row['Filename']
                    if img_path.exists():
                        self.images.append(str(img_path))
                        self.labels.append(int(row['ClassId']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ResnetClassifier(TrafficSignClassifier):
    """Lightweight ResNet-18 classifier optimized for speed and accuracy."""

    def __init__(self, model_path: str = None, num_classes: int = 43, use_half: bool = True):
        """
        Initialize the ResNet-18 traffic sign classifier.

        Args:
            model_path: Path to the pre-trained model weights
            num_classes: Number of traffic sign classes (43 for GTSRB)
            use_half: Use FP16 half precision for faster inference
        """
        super().__init__(model_path, num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = use_half and self.device == 'cuda'
        self.input_shape = (48, 48)  # Smaller input for speed

        print(f"Using device: {self.device}")
        print(f"Half precision (FP16): {self.use_half}")

        # Define transforms for inference (optimized for speed)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define training transforms (with augmentation for accuracy)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((56, 56)),
            transforms.RandomCrop(self.input_shape),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load or create model
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._create_model()

    def _create_model(self):
        """Create an optimized ResNet-18 model."""
        # Use ResNet-18 for speed
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify final layer for traffic sign classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, self.num_classes)
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Apply optimizations
        self._apply_optimizations()

        print("ResNet-18 model created and optimized")

    def _apply_optimizations(self):
        """Apply optimizations for faster inference."""
        if self.model is None:
            return

        self.model.eval()

        # Enable cudnn benchmarking
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        # Convert to half precision
        if self.use_half:
            self.model.half()
            print("Model converted to FP16 half precision")

        # Warmup
        self._warmup_model()

    def _warmup_model(self):
        """Warmup the model for consistent performance."""
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, *self.input_shape).to(self.device)
        if self.use_half:
            dummy_input = dummy_input.half()

        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        print("Model warmup completed")

    def load_model(self, model_path: str = None) -> None:
        """
        Load the classification model from the specified path.

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

        # Create model architecture
        self._create_model()

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model_path = str(model_path)

        # Apply optimizations
        self._apply_optimizations()

        print(f"Model loaded from {model_path}")

    def train(self, train_data: str, validation_data: str = None,
              epochs: int = 30, batch_size: int = 128, learning_rate: float = 0.001,
              **kwargs) -> Dict[str, Any]:
        """
        Train the ResNet-18 model on GTSRB dataset.

        Args:
            train_data: Path to training data directory
            validation_data: Path to validation data directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training history and metrics
        """
        print("Loading GTSRB dataset...")

        # Load datasets
        train_dataset = GTSRBDataset(train_data, transform=self.train_transform, is_test=False)

        # Split training data for validation if not provided
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        # Update validation dataset transform
        val_dataset.dataset.transform = self.val_transform

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True if self.device == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True if self.device == 'cuda' else False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        # Create fresh model for training (no half precision during training)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, self.num_classes)
        )
        self.model = self.model.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                          patience=3)

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc='Training')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                'acc': f'{100.*train_correct/train_total:.2f}%'})

            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                pbar = tqdm(val_loader, desc='Validation')
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                    pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                    'acc': f'{100.*val_correct/val_total:.2f}%'})

            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Learning rate scheduling
            scheduler.step(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/resnet_classifier/best_model.pth')
                print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")

        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

        # Load best model and apply optimizations
        self.load_model('models/resnet_classifier/best_model.pth')

        return history

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify a single traffic sign image.

        Args:
            image: Input image as numpy array (BGR or RGB format)

        Returns:
            Dictionary containing classification results
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")

        start_time = time.time()

        # Preprocess image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self.use_half:
            input_tensor = input_tensor.half()

        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        inference_time = time.time() - start_time

        class_id = predicted.item()
        confidence_score = confidence.item()

        result = {
            'class_id': class_id,
            'class_name': str(class_id),
            'confidence': confidence_score,
            'probabilities': probabilities.cpu().numpy()[0],
            'inference_time_ms': inference_time * 1000
        }

        return result

    def classify_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a traffic sign from an image file path.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing classification results
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        return self.classify(image)

    def classify_tmp_folder(self, tmp_dir: str = "tmp") -> List[Dict[str, Any]]:
        """
        Classify all images in the tmp folder.

        Args:
            tmp_dir: Path to the tmp directory

        Returns:
            List of classification results for each image
        """
        tmp_path = Path(tmp_dir)
        if not tmp_path.exists():
            raise ValueError(f"Directory {tmp_dir} does not exist")

        results = []
        image_files = list(tmp_path.glob('*.png')) + list(tmp_path.glob('*.jpg')) + \
                     list(tmp_path.glob('*.ppm')) + list(tmp_path.glob('*.jpeg'))

        print(f"Classifying {len(image_files)} images from {tmp_dir}...")

        total_time = 0
        for img_path in tqdm(image_files, desc="Classifying"):
            result = self.classify_from_path(str(img_path))
            result['image_path'] = str(img_path)
            result['image_name'] = img_path.name
            results.append(result)
            total_time += result['inference_time_ms']

        if len(results) > 0:
            avg_time = total_time / len(results)
            print(f"\nAverage inference time: {avg_time:.2f} ms")
            print(f"Total time: {total_time:.2f} ms")

        return results

    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            save_path: Path where the model should be saved
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and metadata
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'class_names': self.class_names
        }

        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def evaluate(self, test_data: str) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Path to test data directory (GTSRB root)

        Returns:
            Dictionary containing evaluation metrics
        """
        print("Loading test dataset...")
        test_dataset = GTSRBDataset(test_data, transform=self.val_transform, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.use_half:
                    inputs = inputs.half()

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1

                pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})

        accuracy = 100. * correct / total

        # Calculate per-class accuracy
        per_class_acc = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                per_class_acc[i] = 100. * class_correct[i] / class_total[i]

        print(f"\nTest Accuracy: {accuracy:.2f}%")
        print(f"Correct: {correct}/{total}")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_class_accuracy': per_class_acc
        }
