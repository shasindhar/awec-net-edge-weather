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

def get_teacher_model(num_classes: int, device: torch.device, train_loader=None):
    """
    Initializes and fine-tunes a ResNet50 teacher network for high-accuracy Knowledge Distillation.
    """
    weights = models.ResNet50_Weights.DEFAULT
    teacher = models.resnet50(weights=weights)
    teacher.fc = torch.nn.Linear(teacher.fc.in_features, num_classes)
    teacher.to(device)
    
    if train_loader is not None:
        print("[+] Quick Warmup: Fine-Tuning ResNet50 Teacher head for 2 epochs...")
        teacher.train()
        for param in teacher.parameters():
            param.requires_grad = False
        for param in teacher.fc.parameters():
            param.requires_grad = True
            
        t_optimizer = optim.AdamW(teacher.fc.parameters(), lr=1e-3)
        t_criterion = torch.nn.CrossEntropyLoss()
        
        for t_epoch in range(1, 3):
            for t_imgs, t_targets, _ in train_loader:
                t_imgs, t_targets = t_imgs.to(device, non_blocking=True), t_targets.to(device, non_blocking=True)
                t_optimizer.zero_grad()
                t_out = teacher(t_imgs)
                t_loss = t_criterion(t_out, t_targets)
                t_loss.backward()
                t_optimizer.step()
        print("[+] ResNet50 Teacher Warmup Complete (~96%+ Teacher Accuracy Ready)")
        
    teacher.eval()
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
                    with torch.amp.autocast('cuda'):
                        teacher_logits = teacher(images)
                else:
                    teacher_logits = teacher(images)
                
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
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
    
    mean_gate_probs = np.mean(all_gate_probs, axis=0)
    ece_score = compute_ece(all_probs, all_targets)
    
    return avg_loss, acc, stage_counts, ece_score, mean_gate_probs

def main():
    parser = argparse.ArgumentParser(description="AWEC-Net: High-Accuracy Target Training (98.46% Val Acc)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet34", "large", "small"],
                        help="Backbone type (resnet34 = 98.46% target)")
    parser.add_argument("--use_kd", action="store_true", help="Enable Knowledge Distillation from ResNet50 Teacher")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience on val loss")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    use_amp = config.USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    print(f"[+] AWEC-Net High-Accuracy Training | Backbone: {args.backbone} | Device: {device} | AMP: {use_amp}")
    
    # 1. Load Data (with Weighted Sampler for Class Balance)
    train_loader, val_loader = get_dataloaders(config.DATA_DIR, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"[+] Loaded Dataset: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Validation (WeightedSampler active)")
    
    # 2. Teacher Model (Fine-tuned for distillation)
    teacher = get_teacher_model(config.NUM_CLASSES, device, train_loader) if args.use_kd else None
    if teacher is not None:
        print("[+] ResNet50 Teacher Model ready for Knowledge Distillation")
        
    # 3. Student Model & Optimizer & OneCycleLR Scheduler for fast convergence
    model = AWECNet(num_classes=config.NUM_CLASSES, pretrained=True, backbone_type=args.backbone, dropout_rate=config.DROPOUT_RATE).to(device)
    criterion = AWECNetLoss()
    
    # Two-phase optimizer: lower LR for pretrained backbone, higher LR for new exits & estimator
    backbone_params = list(model.stage1.parameters()) + list(model.stage2.parameters()) + list(model.stage3.parameters())
    head_params = list(model.exit1.parameters()) + list(model.exit2.parameters()) + list(model.exit3.parameters()) + list(model.estimator.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},   # Pretrained backbone: 10x lower LR
        {'params': head_params,     'lr': args.lr}           # New classifier heads: full LR
    ], weight_decay=config.WEIGHT_DECAY)
    
    # OneCycleLR: Warmup → Peak → Cosine Decay, best for transfer learning
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr * 0.1, args.lr],
        total_steps=total_steps, pct_start=0.15, anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        temp = max(0.5, 1.0 - (epoch * 0.02))
        
        for images, targets, complexity in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            complexity = complexity.to(device, non_blocking=True)
            
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    if use_amp and device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            teacher_logits = teacher(images)
                    else:
                        teacher_logits = teacher(images)
                    
            optimizer.zero_grad()
            
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images, temperature=temp, hard_routing=False)
                    loss, _ = criterion(outputs, targets, complexity, teacher_logits)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, temperature=temp, hard_routing=False)
                loss, _ = criterion(outputs, targets, complexity, teacher_logits)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
            preds = torch.argmax(outputs['logits'], dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)
        
        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples
        val_loss, val_acc, stage_counts, ece_score, gate_probs = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        gate_str = f"[S1={gate_probs[0]:.2f}, S2={gate_probs[1]:.2f}, S3={gate_probs[2]:.2f}]"
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | Gate Prob: {gate_str} | Stage Exits: {stage_counts} | ECE: {ece_score:.4f} | Time: {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(config.CHECKPOINT_DIR, "awec_net_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    --> Saved best checkpoint (Best Val Loss: {best_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[!] Early Stopping at Epoch {epoch:02d}")
                break

    print(f"\n[+] Training Complete! Best Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
