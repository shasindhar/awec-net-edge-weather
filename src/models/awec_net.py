import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import torchvision.models as models
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
    def __init__(self, num_classes: int = 5, pretrained: bool = True, embed_dims: Tuple[int, int, int] = (24, 48, 576), dropout_rate: float = 0.3):
        super(AWECNet, self).__init__()
        self.num_classes = num_classes
        
        # 1. Complexity Estimator
        self.estimator = VisualComplexityEstimator(num_exits=3)
        
        if pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                base_model = models.mobilenet_v3_small(weights=weights)
            except Exception:
                base_model = models.mobilenet_v3_small(pretrained=True)
                
            feats = base_model.features
            self.stage1 = feats[0:3]   # Output: 24 channels
            self.stage2 = feats[3:8]   # Output: 48 channels
            self.stage3 = feats[8:13]  # Output: 576 channels
            c1, c2, c3 = 24, 48, 576
        else:
            c1, c2, c3 = embed_dims
            self.stage1 = nn.Sequential(
                nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Hardswish(inplace=True),
                nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Hardswish(inplace=True),
                nn.Dropout2d(0.1)
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.Hardswish(inplace=True),
                nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Hardswish(inplace=True),
                nn.Dropout2d(0.15)
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.Hardswish(inplace=True),
                nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, groups=c3, bias=False),
                nn.BatchNorm2d(c3),
                nn.Hardswish(inplace=True),
                nn.Dropout2d(0.2)
            )
            
        self.exit1 = EarlyExitClassifier(c1, num_classes, dropout_rate=dropout_rate)
        self.exit2 = EarlyExitClassifier(c2, num_classes, dropout_rate=dropout_rate)
        self.exit3 = EarlyExitClassifier(c3, num_classes, dropout_rate=dropout_rate)

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
