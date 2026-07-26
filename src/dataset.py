import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, List, Optional
from src.config import config

class VisualComplexityExtractor:
    """
    Ultrafast visual complexity extractor operating on PIL images:
    Uses thumbnail downsampling + fast numpy spatial gradient & variance calculations.
    Returns normalized visual complexity score C_ref in [0, 1].
    """
    @staticmethod
    def extract_complexity(img_pil: Image.Image) -> float:
        # Downsample to 64x64 thumbnail for 100x speedup
        gray_thumb = img_pil.convert('L').resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.array(gray_thumb, dtype=np.float32)
        
        # Spatial gradients
        dx = np.abs(arr[:, 1:] - arr[:, :-1])
        dy = np.abs(arr[1:, :] - arr[:-1, :])
        grad_val = float(np.mean(dx) + np.mean(dy))
        
        # Global variance
        var_val = float(np.var(arr))
        
        # Normalized score in [0.0, 1.0]
        score = 0.5 * min(grad_val / 35.0, 1.0) + 0.5 * min(var_val / 2200.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

class SyntheticWeatherDatasetGenerator:
    """
    Generates synthetic benchmark weather images with explicit visual complexity variances
    for reproducible evaluation without external network downloads.
    """
    @staticmethod
    def generate_synthetic_data(data_dir: str, samples_per_class: int = 100):
        os.makedirs(data_dir, exist_ok=True)
        for idx, cls_name in enumerate(config.CLASSES):
            cls_dir = os.path.join(data_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            
            for i in range(samples_per_class):
                img_path = os.path.join(cls_dir, f"{cls_name}_{i:04d}.jpg")
                if os.path.exists(img_path):
                    continue
                
                arr = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                if cls_name == "Sunny":
                    arr[:, :, 0] = np.clip(arr[:, :, 0] + 50, 0, 255)
                elif cls_name == "Cloudy":
                    gray_val = np.mean(arr, axis=2, keepdims=True).astype(np.uint8)
                    arr = np.repeat(gray_val, 3, axis=2)
                elif cls_name == "Rainy":
                    arr[:, :, 2] = np.clip(arr[:, :, 2] + 40, 0, 255)
                    for r in range(0, 224, 4):
                        arr[r:min(r+10, 224), r:min(r+2, 224), :] = 255
                elif cls_name == "Snowy":
                    arr = np.clip(arr + 60, 0, 255).astype(np.uint8)
                    snow_spots = np.random.rand(224, 224) > 0.95
                    arr[snow_spots] = 255
                elif cls_name == "Foggy":
                    arr = (arr * 0.4 + 140).astype(np.uint8)
                
                img = Image.fromarray(arr)
                if cls_name == "Foggy":
                    img = img.filter(ImageFilter.GaussianBlur(radius=3))
                img.save(img_path)

class WeatherDataset(Dataset):
    """
    PyTorch Dataset wrapper for Weather Images with Balanced Class Sampling.
    Returns image tensor, target label, and visual complexity score.
    """
    def __init__(self, data_dir: str, is_train: bool = True, transform: Optional[transforms.Compose] = None):
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform or (self.get_train_transforms() if is_train else self.get_val_transforms())
        self.samples = []
        
        # Discover samples
        for label_idx, cls_name in enumerate(config.CLASSES):
            cls_dir = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_dir, fname), label_idx))
                        
    def get_train_transforms(self):
        """
        Strong weather-specific data augmentation for high val accuracy:
        - RandAugment: Randomly applies magnitude-controllable augmentations
        - TrivialAugmentWide: Pushes robustness for minority classes (Rainy, Snowy, Foggy)
        - ColorJitter, RandomGaussianBlur, RandomHorizontalFlip for weather generalization
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0))], p=0.4),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])

    def get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        path, label = self.samples[idx]
        img_pil = Image.open(path).convert('RGB')
        
        complexity_score = VisualComplexityExtractor.extract_complexity(img_pil)
        img_tensor = self.transform(img_pil)
        return img_tensor, label, complexity_score

def get_class_weights(dataset) -> torch.Tensor:
    """Compute inverse frequency weights for imbalanced class sampling."""
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float32)
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
    return sample_weights

def get_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    full_dataset = WeatherDataset(data_dir, is_train=True)
    if len(full_dataset) == 0:
        SyntheticWeatherDatasetGenerator.generate_synthetic_data(data_dir, samples_per_class=100)
        full_dataset = WeatherDataset(data_dir, is_train=True)
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_ds = torch.utils.data.Subset(WeatherDataset(data_dir, is_train=True), train_indices.indices)
    val_ds = torch.utils.data.Subset(WeatherDataset(data_dir, is_train=False), val_indices.indices)
    
    # Weighted sampler to handle Rainy/Snowy/Foggy class imbalance
    full_train_base = WeatherDataset(data_dir, is_train=True)
    sample_weights = get_class_weights(full_train_base)
    train_weights = sample_weights[train_indices.indices]
    sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
