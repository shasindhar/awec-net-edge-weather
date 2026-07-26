import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.config import config
from src.dataset import get_dataloaders
from src.models.awec_net import AWECNet
from src.loss import AWECNetLoss
from src.calibration import compute_ece

def run_ablation_experiment(
    exp_name: str,
    use_wcem: bool,
    use_kd: bool,
    pretrained: bool,
    epochs: int,
    train_loader,
    val_loader,
    device
) -> dict:
    print(f"\n{'='*70}")
    print(f"  Ablation Experiment: {exp_name}")
    print(f"{'='*70}")

    model = AWECNet(
        num_classes=config.NUM_CLASSES,
        pretrained=pretrained,
        backbone_type="resnet34",
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    criterion = AWECNetLoss(
        alpha_kd=0.3 if use_kd else 0.0,
        lambda_route=0.20 if use_wcem else 0.0
    )

    # Two-phase optimizer same as train.py
    backbone_params = (
        list(model.stage1.parameters()) +
        list(model.stage2.parameters()) +
        list(model.stage3.parameters())
    )
    head_params = (
        list(model.exit1.parameters()) +
        list(model.exit2.parameters()) +
        list(model.exit3.parameters()) +
        list(model.estimator.parameters())
    )
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1},
        {'params': head_params,     'lr': config.LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)

    use_amp = config.USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Teacher for KD
    teacher = None
    if use_kd:
        import torchvision.models as tv_models
        try:
            teacher = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        except Exception:
            teacher = tv_models.resnet50(pretrained=True)
        teacher.fc = nn.Linear(teacher.fc.in_features, config.NUM_CLASSES)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  [+] ResNet50 Teacher loaded for KD")

    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[config.LEARNING_RATE * 0.1, config.LEARNING_RATE],
        total_steps=total_steps, pct_start=0.15, anneal_strategy='cos'
    )

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_correct, epoch_total = 0, 0

        for images, targets, complexity in train_loader:
            images   = images.to(device, non_blocking=True)
            targets  = targets.to(device, non_blocking=True)
            complexity = complexity.to(device, non_blocking=True)

            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            teacher_logits = teacher(images)
                    else:
                        teacher_logits = teacher(images)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images, hard_routing=False)
                    if not use_wcem:
                        B = images.size(0)
                        outputs['routing_weights'] = torch.zeros(B, 3, device=device)
                        outputs['routing_weights'][:, 2] = 1.0
                        outputs['logits'] = outputs['out3']
                    loss, _ = criterion(outputs, targets, complexity, teacher_logits)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, hard_routing=False)
                if not use_wcem:
                    B = images.size(0)
                    outputs['routing_weights'] = torch.zeros(B, 3, device=device)
                    outputs['routing_weights'][:, 2] = 1.0
                    outputs['logits'] = outputs['out3']
                loss, _ = criterion(outputs, targets, complexity, teacher_logits)
                loss.backward()
                optimizer.step()

            scheduler.step()
            preds = torch.argmax(outputs['logits'], dim=1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total   += targets.size(0)

        train_acc = epoch_correct / epoch_total
        print(f"  Epoch [{epoch:02d}/{epochs}] Train Acc: {train_acc*100:.2f}%")

    train_time = time.time() - t_start

    # ── Evaluation ──────────────────────────────────────────────────────────
    model.eval()
    correct, total = 0, 0
    all_probs, all_targets = [], []
    stage_counts = [0, 0, 0]

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images, hard_routing=True)
            if not use_wcem:
                outputs['logits'] = outputs['out3']
                outputs['routing_weights'] = torch.zeros(
                    images.size(0), 3, device=device)
                outputs['routing_weights'][:, 2] = 1.0

            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            routes = torch.argmax(outputs['routing_weights'], dim=1)
            for r in routes:
                stage_counts[r.item()] += 1

    acc = round((correct / total) * 100.0, 2) if total > 0 else 0.0
    all_probs   = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    ece = round(compute_ece(all_probs, all_targets), 4)

    print(f"  --> Val Accuracy: {acc:.2f}% | ECE: {ece:.4f} | Stage Exits: {stage_counts} | Train Time: {train_time:.1f}s")

    return {
        "Ablation Setting":      exp_name,
        "Pretrained Backbone":   "Yes" if pretrained else "No",
        "WCEM Dynamic Gate":     "Yes" if use_wcem   else "No (Stage 3 Only)",
        "Knowledge Distillation":"Yes" if use_kd     else "No",
        "Val Accuracy (%)":      acc,
        "ECE":                   ece,
        "Stage Exits (S1/S2/S3)": f"{stage_counts[0]}/{stage_counts[1]}/{stage_counts[2]}",
        "Train Time (s)":        round(train_time, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="AWEC-Net Ablation Study (4 configs)")
    parser.add_argument("--epochs",     type=int, default=5,  help="Epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n[+] AWEC-Net Ablation Study | Device: {device} | Epochs: {args.epochs}")
    train_loader, val_loader = get_dataloaders(
        config.DATA_DIR, batch_size=args.batch_size, num_workers=0)
    print(f"[+] Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")

    # ── 4-Variant Ablation Table ─────────────────────────────────────────────
    experiments = [
        # (name,                            wcem,  kd,    pretrained)
        ("w/o Pretrained + w/o WCEM + w/o KD",  False, False, False),
        ("Pretrained + w/o WCEM + w/o KD",       False, False, True),
        ("Pretrained + WCEM + w/o KD",            True,  False, True),
        ("Full AWEC-Net (Pretrained+WCEM+KD)",    True,  True,  True),
    ]

    results = []
    for exp_name, use_wcem, use_kd, pretrained in experiments:
        res = run_ablation_experiment(
            exp_name, use_wcem, use_kd, pretrained,
            args.epochs, train_loader, val_loader, device)
        results.append(res)

    df = pd.DataFrame(results)
    print("\n" + "="*90)
    print("                        AWEC-Net ABLATION STUDY RESULTS")
    print("="*90)
    print(df.to_string(index=False))

    os.makedirs("./logs", exist_ok=True)
    report_path = "./logs/ablation_results.csv"
    df.to_csv(report_path, index=False)
    print(f"\n[+] Ablation results saved → {report_path}")
    print("="*90)

if __name__ == "__main__":
    main()
