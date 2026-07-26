import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    # Dataset Config
    CLASSES: List[str] = None
    NUM_CLASSES: int = 5
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    DATA_DIR: str = "./data/weather_dataset"
    CHECKPOINT_DIR: str = "./checkpoints"
    LOG_DIR: str = "./logs"
    
    # Model Architecture Config
    BACKBONE_NAME: str = "resnet34"
    GATE_HIDDEN_DIM: int = 64
    STAGE_CHANNELS: Tuple[int, int, int] = (64, 256, 512)
    
    # Adaptive Thresholds & Dynamic Routing Config
    EXIT_THRESHOLDS: Tuple[float, float] = (0.85, 0.90)
    COMPLEXITY_BOUNDS: Tuple[float, float] = (0.35, 0.65)  # Balanced Low/Mid/High split
    LAMBDA_ROUTE_WEIGHT: float = 0.20   # Reduced to let classification lead
    COMPLEXITY_PENALTY_WEIGHT: float = 0.10
    
    # Training & Regularization Config
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 0
    EPOCHS: int = 20
    LEARNING_RATE: float = 5e-4          # Lower LR for ResNet34 fine-tuning
    WEIGHT_DECAY: float = 1e-4           # Reduced weight decay to allow faster learning
    DROPOUT_RATE: float = 0.2            # Lighter dropout for 512-ch heads
    PATIENCE: int = 7                    # Early stopping patience
    TEMPERATURE: float = 1.0             # Gumbel-softmax initial temperature
    USE_AMP: bool = True                 # Automatic Mixed Precision for faster GPU training
    
    def __post_init__(self):
        if self.CLASSES is None:
            self.CLASSES = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

config = Config()
