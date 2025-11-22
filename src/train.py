# train.py
print("Starting train.py...")
import os
print("Imported os")
import argparse
print("Imported argparse")
import time
print("Imported time")
import numpy as np
print("Imported numpy")
from collections import Counter
print("Imported collections")

import torch
print("Imported torch")
import torch.nn as nn
print("Imported torch.nn")
import torch.optim as optim
print("Imported torch.optim")
from torch.utils.data import DataLoader, WeightedRandomSampler
print("Imported torch.utils.data")

from dataset import get_transforms, BrainMRIDataset, make_file_lists
print("Imported dataset")
from model import create_model
print("Imported model")
from sklearn.metrics import confusion_matrix, classification_report
print("Imported sklearn")

def evaluate_metrics(model, loader, classes, device):
    model.eval()
    preds=[]
    targs=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            targs.extend(yb.cpu().numpy().tolist())
    cm = confusion_matrix(targs, preds, labels=list(range(len(classes))))
    report = classification_report(targs, preds, target_names=classes, digits=4, zero_division=0)
    overall_acc = np.sum(np.diag(cm)) / np.sum(cm)
    return overall_acc, cm, report

def main(args):
    # device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    classes, train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = make_file_lists(
        args.train_dir, args.test_dir
    )
    print("Classes:", classes)
    train_transform, val_transform = get_transforms(args.img_size)

    train_ds = BrainMRIDataset(train_paths, train_labels, transform=train_transform)
    val_ds   = BrainMRIDataset(val_paths, val_labels, transform=val_transform)
    test_ds  = BrainMRIDataset(test_paths, test_labels, transform=val_transform)

    # weighted sampler
    counter = Counter(train_labels)
    print("Train class distribution:", counter)
    total = len(train_ds)
    class_sample_counts = [counter[i] for i in range(len(classes))]
    weights_per_class = [ total / (len(classes) * class_sample_counts[i]) for i in range(len(classes))]
    example_weights = [weights_per_class[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=example_weights, num_samples=len(example_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = create_model(num_classes=len(classes), use_qnn=args.use_qnn)
    model = model.to(device)
    # class weights (alternative to sampler)
    # class weights (alternative to sampler) - REMOVED to avoid double weighting with sampler
    # class_counts = [counter[i] for i in range(len(classes))]
    # class_weights = torch.tensor([ total / (len(classes) * c) for c in class_counts ], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # --- Robust AMP/scaler setup for multiple PyTorch versions ---
    scaler = None
    use_amp = False
    if device.type == "cuda":
        use_amp = True
        # Try the newest API first, then fall back to older cuda.amp API
        try:
            # New API (PyTorch 2.1+)
            scaler = torch.amp.GradScaler(device_type='cuda')
        except TypeError:
            try:
                # Older API
                scaler = torch.cuda.amp.GradScaler()
            except Exception:
                scaler = None
                use_amp = False
    else:
        # On CPU: don't use AMP
        scaler = None
        use_amp = False
    # --- end robust AMP setup ---

    best_val_macro = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "best.pth")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses=[]
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                # Mixed precision path
                try:
                    # New autocast API (PyTorch 2.1+)
                    with torch.amp.autocast(device_type=device_type):
                        out = model(xb)
                        loss = criterion(out, yb)
                except TypeError:
                    # Fallback to older cuda.amp.autocast signature
                    with torch.cuda.amp.autocast():
                        out = model(xb)
                        loss = criterion(out, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # FP32 fallback (CPU or no scaler available)
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))
        avg_loss = np.mean(losses)

        val_acc, val_cm, val_report = evaluate_metrics(model, val_loader, classes, device)
        per_class_recall = np.diag(val_cm) / (val_cm.sum(axis=1) + 1e-8)
        val_macro_recall = per_class_recall.mean()

        # step scheduler and print lr change
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"LR changed: {old_lr:.6e} -> {new_lr:.6e}")

        t1 = time.time()
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f} val_macro_recall={val_macro_recall:.4f} time={t1-t0:.1f}s")
        print("Per-class recall:", dict(zip(classes, [float(x) for x in per_class_recall])))
        print("Val classification report:\n", val_report)

        if val_macro_recall > best_val_macro:
            best_val_macro = val_macro_recall
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes': classes,
                'val_acc': val_acc,
                'val_macro_recall': val_macro_recall
            }, ckpt_path)
            print("Saved best model:", ckpt_path)

    print("Training complete. Best val macro recall:", best_val_macro)

    # load best and evaluate on test
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Loaded best checkpoint for test eval.")
    test_acc, test_cm, test_report = evaluate_metrics(model, test_loader, classes, device)
    print("Test overall acc:", test_acc)
    print("Confusion matrix:\n", test_cm)
    print("Classification report:\n", test_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="path to training folder (contains class subfolders)")
    parser.add_argument("--test_dir", type=str, required=True, help="path to testing folder (contains class subfolders)")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="where to save best model")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_qnn", action="store_true", help="Use Quantum Neural Network layer")
    args = parser.parse_args()
    main(args)
