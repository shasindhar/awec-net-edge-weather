import torch
import torch.nn as nn
import torch.nn.functional as F

class AWECNetLoss(nn.Module):
    """
    Dual-Objective Adaptive Compression Loss Function for AWEC-Net.
    
    L_total = L_cls + \lambda_cost * L_cost + \lambda_complexity * L_align
    
    where:
    - L_cls: Multi-exit Cross Entropy classification loss
    - L_cost: FLOPs computational cost penalty (penalizes deep exit routing when unnecessary)
    - L_align: Alignment loss between predicted visual complexity score C_\phi(x) and ground-truth complexity
    """
    def __init__(self, stage_costs: Tuple[float, float, float] = (0.2, 0.5, 1.0), lambda_cost: float = 0.35, lambda_align: float = 0.15):
        super(AWECNetLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.register_buffer('stage_costs', torch.tensor(stage_costs, dtype=torch.float32))
        self.lambda_cost = lambda_cost
        self.lambda_align = lambda_align

    def forward(self, outputs: dict, targets: torch.Tensor, ground_truth_complexity: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 1. Multi-exit Classification Loss
        l1 = self.ce(outputs['out1'], targets)
        l2 = self.ce(outputs['out2'], targets)
        l3 = self.ce(outputs['out3'], targets)
        l_adaptive = self.ce(outputs['logits'], targets)
        
        l_cls = l_adaptive + 0.3 * (l1 + l2 + l3)
        
        # 2. Computational Cost Loss (Expected relative FLOPs)
        # routing_weights: (B, 3)
        routing_weights = outputs['routing_weights']
        stage_costs = self.stage_costs.to(routing_weights.device)
        expected_cost = torch.sum(routing_weights * stage_costs, dim=1) # (B,)
        
        # Encourage cost to align with image visual complexity
        # High complexity -> higher allowed cost; Low complexity -> low cost penalty
        gt_comp = ground_truth_complexity.view(-1).float()
        l_cost = torch.mean(F.relu(expected_cost - gt_comp))
        
        # 3. Visual Complexity Estimator Alignment Loss
        pred_comp = outputs['complexity_score'].view(-1).float()
        l_align = self.mse(pred_comp, gt_comp)
        
        total_loss = l_cls + self.lambda_cost * l_cost + self.lambda_align * l_align
        
        metrics = {
            'total_loss': total_loss.item(),
            'l_cls': l_cls.item(),
            'l_cost': l_cost.item(),
            'l_align': l_align.item(),
            'avg_cost_ratio': expected_cost.mean().item()
        }
        return total_loss, metrics
