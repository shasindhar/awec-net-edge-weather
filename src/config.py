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
    BACKBONE_NAME: str = "mobilenetv3_small"
    GATE_HIDDEN_DIM: int = 64
    STAGE_CHANNELS: Tuple[int, int, int] = (16, 48, 96)
    
    # Adaptive Thresholds & Dynamic Routing Config
    EXIT_THRESHOLDS: Tuple[float, float] = (0.85, 0.90)
    COMPLEXITY_BOUNDS: Tuple[float, float] = (0.25, 0.70)  # Low: <0.25, Mid: 0.25-0.70, High: >0.70
    LAMBDA_ROUTE_WEIGHT: float = 0.40  # Routing alignment loss weight
    COMPLEXITY_PENALTY_WEIGHT: float = 0.20  # Lambda factor in dual-objective loss
    
    # Training & Regularization Config
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0
    EPOCHS: int = 30
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-3  # Increased weight decay to combat overfitting
    DROPOUT_RATE: float = 0.3   # Increased dropout rate for classifier heads & backbone
    PATIENCE: int = 6           # Early stopping patience on validation loss
    TEMPERATURE: float = 1.0    # Gumbel-softmax initial temperature
    USE_AMP: bool = True        # Automatic Mixed Precision for faster GPU training
    
    def __post_init__(self):
        if self.CLASSES is None:
            self.CLASSES = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

config = Config()
