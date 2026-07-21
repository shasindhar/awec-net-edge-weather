import os
import time
import argparse
import torch
import torch.optim as optim
import torchvision.models as models
from src.config import config
from src.dataset import get_dataloaders
from src.models.awec_net import AWECNet
from src.loss import AWECNetLoss
from src.calibration import compute_ece
import numpy as np

def get_teacher_model(num_classes: int, device: torch.device):
    """
    Initializes a ResNet50 teacher network for Knowledge Distillation.
    """
    weights = models.ResNet50_Weights.DEFAULT
    teacher = models.resnet50(weights=weights)
    teacher.fc = torch.nn.Linear(teacher.fc.in_features, num_classes)
    teacher.to(device)
    teacher.eval() # Teacher is kept frozen
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

def train_one_epoch(model, teacher, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    temp = max(0.5, 1.0 - (epoch * 0.02)) # Temperature annealing for Gumbel-Softmax
    
    for images, targets, complexity in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        complexity = complexity.to(device)
        
        # Compute teacher soft logits if teacher enabled
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(images)
                
        optimizer.zero_grad()
        outputs = model(images, temperature=temp, hard_routing=False)
        loss, loss_metrics = criterion(outputs, targets, complexity, teacher_logits)
        
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs['logits'], dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
        total_loss += loss.item() * targets.size(0)
        
    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_loss, acc

def evaluate(model, dataloader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    stage_counts = [0, 0, 0]
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images, hard_routing=True)
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            
            routes = torch.argmax(outputs['routing_weights'], dim=1)
            for r in routes:
                stage_counts[r.item()] += 1
                
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
                
    acc = total_correct / total_samples
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    ece_score = compute_ece(all_probs, all_targets)
    
    return acc, stage_counts, ece_score

def main():
    parser = argparse.ArgumentParser(description="Train AWEC-Net Weather Classifier with KD")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--use_kd", action="store_true", help="Enable Knowledge Distillation from ResNet50 Teacher")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Starting AWEC-Net Training on Device: {device}")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(config.DATA_DIR, batch_size=args.batch_size)
    print(f"[+] Loaded Dataset: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Validation")
    
    # 2. Teacher Model
    teacher = get_teacher_model(config.NUM_CLASSES, device) if args.use_kd else None
    if teacher is not None:
        print("[+] ResNet50 Teacher Model initialized for Knowledge Distillation")
        
    # 3. Student Model & Optimizer
    model = AWECNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = AWECNetLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, teacher, train_loader, criterion, optimizer, device, epoch)
        val_acc, stage_counts, ece_score = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | ECE: {ece_score:.4f} | Stage Exits: {stage_counts} | Time: {elapsed:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.CHECKPOINT_DIR, "awec_net_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    --> Saved best checkpoint to {save_path}")

if __name__ == "__main__":
    main()
