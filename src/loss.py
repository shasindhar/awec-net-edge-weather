import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class AWECNetLoss(nn.Module):
    r"""
    Dual-Objective Adaptive Compression Loss Function with Knowledge Distillation (KD).
    
    L_total = L_cls + \alpha_kd * L_kd + \lambda_cost * L_cost + \lambda_align * L_align
    
    where:
    - L_cls: Multi-exit Cross Entropy classification loss
    - L_kd: Knowledge Distillation KL-Divergence loss between Teacher soft logits & Student exits
    - L_cost: FLOPs computational cost penalty (penalizes deep exit routing when unnecessary)
    - L_align: Alignment loss between predicted visual complexity score C_\phi(x) and ground-truth complexity
    """
    def __init__(self, stage_costs: Tuple[float, float, float] = (0.2, 0.5, 1.0), lambda_cost: float = 0.35, lambda_align: float = 0.15, alpha_kd: float = 0.4, kd_temperature: float = 3.0):
        super(AWECNetLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.register_buffer('stage_costs', torch.tensor(stage_costs, dtype=torch.float32))
        self.lambda_cost = lambda_cost
        self.lambda_align = lambda_align
        self.alpha_kd = alpha_kd
        self.kd_temperature = kd_temperature

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, ground_truth_complexity: torch.Tensor, teacher_logits: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        # 1. Multi-exit Classification Loss
        l1 = self.ce(outputs['out1'], targets)
        l2 = self.ce(outputs['out2'], targets)
        l3 = self.ce(outputs['out3'], targets)
        l_adaptive = self.ce(outputs['logits'], targets)
        
        l_cls = l_adaptive + 0.3 * (l1 + l2 + l3)
        
        # 2. Knowledge Distillation Loss (if teacher logits provided)
        l_kd = torch.tensor(0.0, device=targets.device)
        if teacher_logits is not None:
            T = self.kd_temperature
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)
            
            kd1 = self.kld(F.log_softmax(outputs['out1'] / T, dim=-1), teacher_soft) * (T ** 2)
            kd2 = self.kld(F.log_softmax(outputs['out2'] / T, dim=-1), teacher_soft) * (T ** 2)
            kd3 = self.kld(F.log_softmax(outputs['out3'] / T, dim=-1), teacher_soft) * (T ** 2)
            l_kd = (kd1 + kd2 + kd3) / 3.0
            
        # 3. Computational Cost Loss (Expected relative FLOPs)
        routing_weights = outputs['routing_weights']
        stage_costs = self.stage_costs.to(routing_weights.device)
        expected_cost = torch.sum(routing_weights * stage_costs, dim=1) # (B,)
        
        gt_comp = ground_truth_complexity.view(-1).float()
        l_cost = torch.mean(F.relu(expected_cost - gt_comp))
        
        # 4. Visual Complexity Estimator Alignment Loss
        pred_comp = outputs['complexity_score'].view(-1).float()
        l_align = self.mse(pred_comp, gt_comp)
        
        total_loss = l_cls + self.alpha_kd * l_kd + self.lambda_cost * l_cost + self.lambda_align * l_align
        
        metrics = {
            'total_loss': total_loss.item(),
            'l_cls': l_cls.item(),
            'l_kd': l_kd.item(),
            'l_cost': l_cost.item(),
            'l_align': l_align.item(),
            'avg_cost_ratio': expected_cost.mean().item()
        }
        return total_loss, metrics
