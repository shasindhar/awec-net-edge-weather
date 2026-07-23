import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualComplexityEstimator(nn.Module):
    r"""
    Lightweight Visual Complexity Estimator G_\phi(x).
    Consists of ultralight depthwise-separable convolutions and a global average pool.
    Input: (B, 3, H, W) weather image tensor
    Output:
        - complexity_score: (B, 1) scalar in [0, 1] representing visual complexity
        - routing_weights: (B, 3) softmax probability distribution for selecting exit stage (Exit 1, Exit 2, Full Backbone)
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 32, num_exits: int = 3):
        super(VisualComplexityEstimator, self).__init__()
        
        # Fast downsampling stream
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, hidden_dim, kernel_size=3, stride=2, padding=1, groups=16)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Heads
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.routing_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_exits)
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard: bool = False):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = F.relu(self.bn3(self.conv3(feat)))
        
        pooled = self.gap(feat).squeeze(-1).squeeze(-1)
        
        complexity_score = self.complexity_head(pooled)  # (B, 1)
        logits = self.routing_head(pooled)               # (B, num_exits)
        
        if self.training:
            # Gumbel-Softmax differentiable routing during training
            routing_weights = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        else:
            # Softmax / Hard decision during evaluation
            if hard:
                idx = torch.argmax(logits, dim=-1)
                routing_weights = F.one_hot(idx, num_classes=logits.size(-1)).float()
            else:
                routing_weights = F.softmax(logits / temperature, dim=-1)
                
        return complexity_score, routing_weights, logits
