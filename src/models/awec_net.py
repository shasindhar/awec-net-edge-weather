import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from src.models.complexity_estimator import VisualComplexityEstimator

class EarlyExitClassifier(nn.Module):
    """
    Lightweight classification head attached to intermediate backbone stages.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.2):
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
    High-Performance Edition with Pretrained MobileNetV3-Large Backbone (>97% Accuracy Target).
    
    Consists of:
    1. G_\phi: Ultralight Visual Complexity Estimator
    2. Stage 1 (Fast Exit for clear sunny / low complexity images)
    3. Stage 2 (Medium Exit for moderate overcast images)
    4. Stage 3 (Deep Backbone for high complexity / fog / rain / snow images)
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, backbone_type: str = "large", dropout_rate: float = 0.2):
        super(AWECNet, self).__init__()
        self.num_classes = num_classes
        
        # 1. Complexity Estimator
        self.estimator = VisualComplexityEstimator(num_exits=3)
        
        # 2. Backbone Stages (High-Capacity Pretrained Feature Extractor)
        is_pretrained_loaded = False
        if pretrained:
            try:
                import torchvision.models as models
                if backbone_type == "large":
                    try:
                        weights = models.MobileNet_V3_Large_Weights.DEFAULT
                        base_model = models.mobilenet_v3_large(weights=weights)
                    except Exception:
                        base_model = models.mobilenet_v3_large(pretrained=True)
                    feats = base_model.features
                    self.stage1 = feats[0:4]   # Stage 1 (24 channels)
                    self.stage2 = feats[4:7]   # Stage 2 (40 channels)
                    self.stage3 = feats[7:]    # Stage 3 (960 channels)
                else:
                    try:
                        weights = models.MobileNet_V3_Small_Weights.DEFAULT
                        base_model = models.mobilenet_v3_small(weights=weights)
                    except Exception:
                        base_model = models.mobilenet_v3_small(pretrained=True)
                    feats = base_model.features
                    self.stage1 = feats[0:3]   # Stage 1 (24 channels)
                    self.stage2 = feats[3:8]   # Stage 2 (48 channels)
                    self.stage3 = feats[8:]    # Stage 3 (576 channels)
                is_pretrained_loaded = True
            except Exception:
                is_pretrained_loaded = False

        if not is_pretrained_loaded:
            c1_fallback, c2_fallback, c3_fallback = (24, 48, 96)
            self.stage1 = nn.Sequential(
                nn.Conv2d(3, c1_fallback, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c1_fallback),
                nn.Hardswish(inplace=True),
                nn.Conv2d(c1_fallback, c1_fallback, kernel_size=3, stride=2, padding=1, groups=c1_fallback, bias=False),
                nn.BatchNorm2d(c1_fallback),
                nn.Hardswish(inplace=True),
                nn.Dropout2d(0.1)
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(c1_fallback, c2_fallback, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2_fallback),
                nn.Hardswish(inplace=True),
                nn.Conv2d(c2_fallback, c2_fallback, kernel_size=3, stride=1, padding=1, groups=c2_fallback, bias=False),
                nn.BatchNorm2d(c2_fallback),
                nn.Hardswish(inplace=True),
                nn.Dropout2d(0.15)
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(c2_fallback, c3_fallback, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c3_fallback),
                nn.Hardswish(inplace=True),
                nn.Conv2d(c3_fallback, c3_fallback, kernel_size=3, stride=1, padding=1, groups=c3_fallback, bias=False),
                nn.BatchNorm2d(c3_fallback),
                nn.Hardswish(inplace=True),
                nn.Dropout2d(0.2)
            )

        # Dynamic channel shape detection to guarantee 100% mat1 / mat2 dimension matching
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            f1 = self.stage1(dummy)
            c1 = f1.size(1)
            f2 = self.stage2(f1)
            c2 = f2.size(1)
            f3 = self.stage3(f2)
            c3 = f3.size(1)
            
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
