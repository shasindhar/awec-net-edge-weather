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
    Target Benchmark Edition (98.46% Val Accuracy Target).
    
    Consists of:
    1. G_\phi: Ultralight Visual Complexity Estimator
    2. Stage 1 (Fast Exit for clear sunny / low complexity images, 64 ch)
    3. Stage 2 (Medium Exit for moderate overcast images, 256 ch)
    4. Stage 3 (Deep Backbone for high complexity / fog / rain / snow images, 512 ch)
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, backbone_type: str = "resnet34", dropout_rate: float = 0.2):
        super(AWECNet, self).__init__()
        self.num_classes = num_classes
        
        # 1. Complexity Estimator
        self.estimator = VisualComplexityEstimator(num_exits=3)
        
        # 2. Backbone Stages (ResNet34 / MobileNet Multi-Stage Feature Backbone)
        is_pretrained_loaded = False
        if pretrained:
            try:
                import torchvision.models as models
                if backbone_type == "resnet34":
                    try:
                        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                    except Exception:
                        base = models.resnet34(pretrained=True)
                    
                    self.stage1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1) # 64 ch
                    self.stage2 = nn.Sequential(base.layer2, base.layer3)                                  # 256 ch
                    self.stage3 = nn.Sequential(base.layer4)                                                # 512 ch
                    c1, c2, c3 = 64, 256, 512
                else:
                    try:
                        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
                    except Exception:
                        base = models.mobilenet_v3_large(pretrained=True)
                    feats = base.features
                    self.stage1 = feats[0:4]
                    self.stage2 = feats[4:7]
                    self.stage3 = feats[7:]
                    c1, c2, c3 = 24, 40, 960
                is_pretrained_loaded = True
            except Exception:
                is_pretrained_loaded = False

        if not is_pretrained_loaded:
            c1, c2, c3 = (64, 256, 512)
            self.stage1 = nn.Sequential(
                nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True)
            )

        # Dynamic channel shape detection to guarantee 100% matrix shape matching
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
