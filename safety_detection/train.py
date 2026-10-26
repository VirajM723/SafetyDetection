"""Command-line entry point for training: ``python -m safety_detection.train``."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import create_dataloaders
from .engine import fit, plot_history
from .model import build_model
from .predict import default_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ViT-B/16 for safety gear classification.")
    parser.add_argument("--data-dir", type=Path, default=Path("GearedorNot"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--unfreeze", action="store_true",
                        help="Fine-tune the whole backbone instead of only the head (use a lower --lr, e.g. 1e-5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.val_fraction, args.seed, args.num_workers)
    print(f"Classes: {class_names} | train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)}")

    model = build_model(len(class_names), pretrained=True, freeze_backbone=not args.unfreeze)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_path = args.output_dir / "vit_ppe.pt"
    history = fit(model, train_loader, val_loader, optimizer, loss_fn, args.epochs,
                  device, class_names, checkpoint_path)

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, args.output_dir / "training_curves.png")
    print(f"Best val_acc {max(history['val_acc']):.4f} -> {checkpoint_path}")


if __name__ == "__main__":
    main()
