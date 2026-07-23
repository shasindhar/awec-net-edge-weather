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

def train_one_epoch(model, teacher, dataloader, criterion, optimizer, scaler, device, epoch, use_amp):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    temp = max(0.5, 1.0 - (epoch * 0.02)) # Temperature annealing for Gumbel-Softmax
    
    for images, targets, complexity in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        complexity = complexity.to(device, non_blocking=True)
        
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                if use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        teacher_logits = teacher(images)
                else:
                    teacher_logits = teacher(images)
                
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(images, temperature=temp, hard_routing=False)
                loss, loss_metrics = criterion(outputs, targets, complexity, teacher_logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    stage_counts = [0, 0, 0]
    all_probs, all_targets = [], []
    all_gate_probs = []
    
    with torch.no_grad():
        for images, targets, complexity in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            complexity = complexity.to(device, non_blocking=True)
            
            outputs = model(images, hard_routing=True)
            loss, _ = criterion(outputs, targets, complexity)
            
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)
            
            routes = torch.argmax(outputs['routing_weights'], dim=1)
            for r in routes:
                stage_counts[r.item()] += 1
                
            gate_probs = torch.softmax(outputs['gate_logits'], dim=-1)
            all_gate_probs.append(gate_probs.cpu().numpy())
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
                
    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_gate_probs = np.concatenate(all_gate_probs, axis=0)
    
    mean_gate_probs = np.mean(all_gate_probs, axis=0) # Mean [Stage1, Stage2, Stage3] gate probability
    ece_score = compute_ece(all_probs, all_targets)
    
    return avg_loss, acc, stage_counts, ece_score, mean_gate_probs

def main():
    parser = argparse.ArgumentParser(description="Train AWEC-Net Weather Classifier with KD")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--use_kd", action="store_true", help="Enable Knowledge Distillation from ResNet50 Teacher")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers (set 0 for Google Colab)")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience on val loss")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.USE_AMP and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    print(f"[+] Starting AWEC-Net Training on Device: {device} (AMP Enabled: {use_amp})")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(config.DATA_DIR, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"[+] Loaded Dataset: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Validation")
    
    # 2. Teacher Model
    teacher = get_teacher_model(config.NUM_CLASSES, device) if args.use_kd else None
    if teacher is not None:
        print("[+] ResNet50 Teacher Model initialized for Knowledge Distillation")
        
    # 3. Student Model & Optimizer
    model = AWECNet(num_classes=config.NUM_CLASSES, dropout_rate=config.DROPOUT_RATE).to(device)
    criterion = AWECNetLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, teacher, train_loader, criterion, optimizer, scaler, device, epoch, use_amp)
        val_loss, val_acc, stage_counts, ece_score, gate_probs = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        gate_str = f"[S1={gate_probs[0]:.2f}, S2={gate_probs[1]:.2f}, S3={gate_probs[2]:.2f}]"
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | Gate Prob: {gate_str} | Stage Exits: {stage_counts} | ECE: {ece_score:.4f} | Time: {elapsed:.1f}s")
        
        # Early Stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(config.CHECKPOINT_DIR, "awec_net_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    --> Saved best checkpoint to {save_path} (Best Val Loss: {best_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[!] Early Stopping triggered at Epoch {epoch:02d}: Val loss did not improve for {args.patience} consecutive epochs.")
                break

if __name__ == "__main__":
    main()
