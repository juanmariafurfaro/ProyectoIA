# src/train.py (CPU-friendly)
import os
import argparse
from datetime import datetime
from collections import Counter 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .data.video_dataset import VideoFolderDataset
from .models.cnn_lstm import CNNLSTM
from .utils.seed import set_seed
from .utils.metrics import compute_metrics


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='data', help='Carpeta con subcarpetas por clase')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_frames', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18','resnet34','googlenet'])
    ap.add_argument('--hidden_size', type=int, default=256)
    ap.add_argument('--bidirectional', action='store_true')
    ap.add_argument('--train_backbone', action='store_true')
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str, default='checkpoints')
    # CPU-friendly extras
    ap.add_argument('--num_workers', type=int, default=0)  # 0 en Windows/CPU evita overhead
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--patience', type=int, default=7)     # early stopping
    return ap.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Datasets
    full_ds = VideoFolderDataset(args.data_root, num_frames=args.num_frames, train=True)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # En validación desactivamos augmentations
    val_ds.dataset.train = False
     # ---------- NUEVO: pesos por clase (inverso a la frecuencia) ----------
    # Opción eficiente: usamos los índices del Subset para no cargar todos los videos
    # full_ds.items es una lista de (path, label)
    train_labels = [full_ds.items[i][1] for i in train_ds.indices]
    counts = Counter(train_labels)  # ej: {0: 102, 1: 98}
    total = sum(counts.values())
    num_classes = len(full_ds.classes)

    weights = torch.tensor(
        [total / max(1, counts.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float32
    ).to(device)

    print(f"[Info] Distribución train: {dict(counts)}  |  Pesos CE: {weights.tolist()}")
    # ----------------------------------------------------------------------
    # DataLoaders (pin_memory solo si hay CUDA)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=False
    )

    # Modelo
    model = CNNLSTM(
        num_classes=len(full_ds.classes),
        backbone=args.backbone,
        hidden_size=args.hidden_size,
        bidirectional=args.bidirectional,
        train_backbone=args.train_backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logging
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)

    best_val_acc = 0.0
    best_path = os.path.join(args.out_dir, 'best.pt')
    bad_epochs = 0  # para early stopping

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss = 0.0
        y_true, y_pred = [], []

        for clips, labels in pbar:
            clips = clips.to(device)  # [B,T,C,H,W]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            preds = logits.argmax(dim=1).detach().cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.detach().cpu().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(y_true, y_pred)

        # Validación
        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                labels = labels.to(device)
                logits = model(clips)
                loss = criterion(logits, labels)
                val_loss += loss.item() * clips.size(0)
                preds = logits.argmax(dim=1).detach().cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.detach().cpu().tolist())
        val_loss /= len(val_loader.dataset)
        val_metrics = compute_metrics(y_true, y_pred)

        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_metrics['acc'], epoch)
        writer.add_scalar('Acc/val', val_metrics['acc'], epoch)

        scheduler.step()

        # Early stopping + best ckpt
        improved = val_metrics['acc'] > best_val_acc
        if improved:
            best_val_acc = val_metrics['acc']
            torch.save({
                'model_state': model.state_dict(),
                'args': vars(args),
                'classes': full_ds.classes,
            }, best_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_metrics['acc']:.3f} | val_loss={val_loss:.4f} acc={val_metrics['acc']:.3f} | best_val_acc={best_val_acc:.3f}")

        if bad_epochs >= args.patience:
            print(f"Early stopping activado (paciencia={args.patience}). Mejor val acc: {best_val_acc:.3f}.")
            break

    writer.close()
    print(f"Mejor val acc: {best_val_acc:.3f}. Checkpoint: {best_path}")


if __name__ == '__main__':
    main()