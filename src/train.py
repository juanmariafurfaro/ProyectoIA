import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.video_dataset import VideoFolderDataset
from models.cnn_lstm import CNNLSTM
from utils.seed import set_seed
from utils.metrics import compute_metrics


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='data', help='Carpeta con subcarpetas por clase')
    ap.add_argument('--epochs', type=int, default=20)
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
    ap.add_argument('--mixed_precision', action='store_true')
    ap.add_argument('--out_dir', type=str, default='checkpoints')
    return ap.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Datasets
    full_ds = VideoFolderDataset(args.data_root, num_frames=args.num_frames, train=True)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # En validación desactivamos augmentations
    val_ds.dataset.train = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Modelo
    model = CNNLSTM(num_classes=len(full_ds.classes), backbone=args.backbone,
                    hidden_size=args.hidden_size, bidirectional=args.bidirectional,
                    train_backbone=args.train_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logging
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    best_val_acc = 0.0
    best_path = os.path.join(args.out_dir, 'best.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        y_true, y_pred = [], []

        for clips, labels in pbar:
            clips = clips.to(device)  # [B,T,C,H,W]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        # Guardar mejor
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            torch.save({
                'model_state': model.state_dict(),
                'args': vars(args),
                'classes': full_ds.classes,
            }, best_path)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_metrics['acc']:.3f} | val_loss={val_loss:.4f} acc={val_metrics['acc']:.3f}")

    writer.close()
    print(f"Mejor val acc: {best_val_acc:.3f}. Checkpoint: {best_path}")


if __name__ == '__main__':
    main()