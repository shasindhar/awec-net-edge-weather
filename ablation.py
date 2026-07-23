import os
import argparse
import torch
import numpy as np
import pandas as pd
from src.config import config
from src.dataset import get_dataloaders
from src.models.awec_net import AWECNet
from src.loss import AWECNetLoss
from src.calibration import compute_ece
import torch.optim as optim

def run_ablation_experiment(exp_name: str, use_wcem: bool, use_kd: bool, epochs: int, train_loader, val_loader, device):
    print(f"\n[+] Running Ablation Experiment: {exp_name}")
    
    model = AWECNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = AWECNetLoss(alpha_kd=0.4 if use_kd else 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets, complexity in train_loader:
            images, targets, complexity = images.to(device), targets.to(device), complexity.to(device)
            optimizer.zero_grad()
            
            # Disable WCEM dynamic routing if requested
            outputs = model(images, hard_routing=False)
            if not use_wcem:
                # Force full backbone routing
                outputs['routing_weights'] = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(images.size(0), 1)
                outputs['logits'] = outputs['out3']
                
            loss, _ = criterion(outputs, targets, complexity)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    correct, total = 0, 0
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images, hard_routing=True)
            if not use_wcem:
                outputs['logits'] = outputs['out3']
                
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    acc = round((correct / total) * 100.0, 2)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    ece = round(compute_ece(all_probs, all_targets), 4)
    
    return {
        "Ablation Setting": exp_name,
        "WCEM Dynamic Gate": "Yes" if use_wcem else "No (Stage 3 Only)",
        "Knowledge Distillation": "Yes" if use_kd else "No",
        "Val Accuracy (%)": acc,
        "ECE Calibration Error": ece
    }

def main():
    parser = argparse.ArgumentParser(description="Run AWEC-Net Ablation Study")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per ablation experiment")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(config.DATA_DIR, batch_size=32, num_workers=0)
    
    experiments = [
        ("Static Backbone Alone", False, False),
        ("Backbone + WCEM", True, False),
        ("Backbone + KD", False, True),
        ("Full AWEC-Net (WCEM + KD)", True, True)
    ]
    
    ablation_results = []
    for exp_name, use_wcem, use_kd in experiments:
        res = run_ablation_experiment(exp_name, use_wcem, use_kd, args.epochs, train_loader, val_loader, device)
        ablation_results.append(res)
        
    df = pd.DataFrame(ablation_results)
    print("\n==========================================================================================")
    print("                              AWEC-Net ABLATION STUDY RESULTS                             ")
    print("==========================================================================================")
    print(df.to_string(index=False))
    
    os.makedirs("./logs", exist_ok=True)
    report_path = "./logs/ablation_results.csv"
    df.to_csv(report_path, index=False)
    print(f"\n[+] Ablation results saved to: {report_path}")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
