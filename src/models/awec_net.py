import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from src.models.complexity_estimator import VisualComplexityEstimator

class EarlyExitClassifier(nn.Module):
    """
    Lightweight classification head attached to intermediate backbone stages.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.3):
        super(EarlyExitClassifier, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.gap(x).view(x.size(0), -1)
        return self.fc(feat)

class AWECNet(nn.Module):
    r"""
    AWEC-Net: Weather-Complexity-Aware Adaptive Compression Neural Network.
    
    Consists of:
    1. G_\phi: Ultralight Visual Complexity Estimator
    2. Stage 1 (Fast Exit for low visual complexity/clear sunny images)
    3. Stage 2 (Medium Exit for moderate visual complexity images)
    4. Stage 3 (Full Backbone for high complexity/foggy/blurry/snowy images)
    """
    def __init__(self, num_classes: int = 5, embed_dims: Tuple[int, int, int] = (24, 48, 96), dropout_rate: float = 0.3):
        super(AWECNet, self).__init__()
        self.num_classes = num_classes
        
        # 1. Complexity Estimator
        self.estimator = VisualComplexityEstimator(num_exits=3)
        
        # 2. Stage 1 Blocks (Ultralight Mobile Conv)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.Hardswish(inplace=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=2, padding=1, groups=embed_dims[0], bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.Hardswish(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.exit1 = EarlyExitClassifier(embed_dims[0], num_classes, dropout_rate=dropout_rate)
        
        # 3. Stage 2 Blocks
        self.stage2 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.Hardswish(inplace=True),
            nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1], bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.Hardswish(inplace=True),
            nn.Dropout2d(0.15)
        )
        self.exit2 = EarlyExitClassifier(embed_dims[1], num_classes, dropout_rate=dropout_rate)
        
        # 4. Stage 3 Blocks (Deep Feature Representation)
        self.stage3 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.Hardswish(inplace=True),
            nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2], bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.Hardswish(inplace=True),
            nn.Dropout2d(0.2)
        )
        self.exit3 = EarlyExitClassifier(embed_dims[2], num_classes, dropout_rate=dropout_rate)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard_routing: bool = False) -> Dict[str, torch.Tensor]:
        # Estimate visual complexity score and routing probabilities
        complexity_score, routing_weights, gate_logits = self.estimator(x, temperature=temperature, hard=hard_routing)
        
        # Compute multi-stage outputs
        f1 = self.stage1(x)
        out1 = self.exit1(f1)
        
        f2 = self.stage2(f1)
        out2 = self.exit2(f2)
        
        f3 = self.stage3(f2)
        out3 = self.exit3(f3)
        
        # Combined adaptive output based on dynamic routing weights:
        w1 = routing_weights[:, 0].unsqueeze(1) # (B, 1)
        w2 = routing_weights[:, 1].unsqueeze(1) # (B, 1)
        w3 = routing_weights[:, 2].unsqueeze(1) # (B, 1)
        
        adaptive_logits = w1 * out1 + w2 * out2 + w3 * out3
        
        return {
            'logits': adaptive_logits,
            'out1': out1,
            'out2': out2,
            'out3': out3,
            'complexity_score': complexity_score,
            'routing_weights': routing_weights,
            'gate_logits': gate_logits
        }
