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
    Computes visual complexity features for input images:
    - High-frequency edge energy (Laplacian variance)
    - Spatial intensity gradient magnitude (Sobel energy)
    - Luminance & Color entropy
    Returns a normalized complexity score C_ref in [0, 1].
    """
    @staticmethod
    def extract_complexity(img_pil: Image.Image) -> float:
        # Convert image to grayscale numpy array
        gray = np.array(img_pil.convert('L'), dtype=np.float32)
        
        # 1. Edge Energy (Laplacian variance)
        # Approximate 3x3 Laplacian filter kernel
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        from scipy.ndimage import convolve
        lap_resp = convolve(gray, laplacian_kernel)
        lap_var = float(np.var(lap_resp))
        
        # 2. Intensity Gradient Energy (Sobel)
        sobel_x = convolve(gray, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32))
        sobel_y = convolve(gray, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32))
        grad_mag = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
        
        # 3. Spatial Entropy
        hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize and composite score in [0.0, 1.0]
        score = 0.4 * min(lap_var / 1500.0, 1.0) + 0.4 * min(grad_mag / 80.0, 1.0) + 0.2 * (entropy / 5.0)
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
                
                # Base image with synthetic noise / pattern according to weather type
                arr = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                if cls_name == "Sunny":
                    # Bright, high contrast, warm tone
                    arr[:, :, 0] = np.clip(arr[:, :, 0] + 50, 0, 255)
                elif cls_name == "Cloudy":
                    # Soft gray, low gradient contrast
                    gray_val = np.mean(arr, axis=2, keepdims=True).astype(np.uint8)
                    arr = np.repeat(gray_val, 3, axis=2)
                elif cls_name == "Rainy":
                    # Dark blue tint with high frequency streak lines (rain)
                    arr[:, :, 2] = np.clip(arr[:, :, 2] + 40, 0, 255)
                    # Add diagonal rain streaks
                    for r in range(0, 224, 4):
                        arr[r:min(r+10, 224), r:min(r+2, 224), :] = 255
                elif cls_name == "Snowy":
                    # High brightness snow spots
                    arr = np.clip(arr + 60, 0, 255).astype(np.uint8)
                    snow_spots = np.random.rand(224, 224) > 0.95
                    arr[snow_spots] = 255
                elif cls_name == "Foggy":
                    # Heavy blur, low contrast haze
                    arr = (arr * 0.4 + 140).astype(np.uint8)
                
                img = Image.fromarray(arr)
                if cls_name == "Foggy":
                    img = img.filter(ImageFilter.GaussianBlur(radius=3))
                
                img.save(img_path)

class WeatherDataset(Dataset):
    """
    PyTorch Dataset wrapper for Weather Images. Returns image tensor, target label, and visual complexity score.
    """
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        self.data_dir = data_dir
        self.transform = transform or self.get_default_transforms()
        self.samples = []
        
        # Discover samples
        for label_idx, cls_name in enumerate(config.CLASSES):
            cls_dir = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_dir, fname), label_idx))
                        
    def get_default_transforms(self):
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        path, label = self.samples[idx]
        img_pil = Image.open(path).convert('RGB')
        
        # Calculate ground truth visual complexity score for dynamic routing analysis
        # (Use simple heuristic during dataset loading or pre-computed)
        complexity_score = VisualComplexityExtractor.extract_complexity(img_pil)
        
        img_tensor = self.transform(img_pil)
        return img_tensor, label, complexity_score

def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    dataset = WeatherDataset(data_dir)
    if len(dataset) == 0:
        # Generate synthetic data if empty
        SyntheticWeatherDatasetGenerator.generate_synthetic_data(data_dir, samples_per_class=100)
        dataset = WeatherDataset(data_dir)
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
