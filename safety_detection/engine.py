"""Training and evaluation loops."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import save_checkpoint


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class EvalResult(EpochMetrics):
    # confusion[true][pred]
    confusion: List[List[int]] = field(default_factory=list)


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> EpochMetrics:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weight by batch size so a small final batch doesn't skew the average.
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return EpochMetrics(total_loss / seen, correct / seen)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: torch.device, num_classes: int) -> EvalResult:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        seen += labels.size(0)
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t][p] += 1
    return EvalResult(total_loss / seen, correct / seen, confusion)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int,
        device: torch.device, class_names: List[str],
        checkpoint_path: Optional[str | Path] = None) -> Dict[str, List[float]]:
    """Train for ``epochs`` epochs, keeping the checkpoint with the best validation accuracy."""
    model.to(device)
    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        train = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, loss_fn, device, len(class_names))
        history["train_loss"].append(train.loss)
        history["train_acc"].append(train.accuracy)
        history["val_loss"].append(val.loss)
        history["val_acc"].append(val.accuracy)
        tqdm.write(f"epoch {epoch:>2} | train_loss {train.loss:.4f} | train_acc {train.accuracy:.4f} "
                   f"| val_loss {val.loss:.4f} | val_acc {val.accuracy:.4f}")
        if checkpoint_path and val.accuracy > best_acc:
            best_acc = val.accuracy
            save_checkpoint(checkpoint_path, model, class_names, epoch=epoch, val_acc=val.accuracy)
    return history


def plot_history(history: Dict[str, List[float]], save_path: Optional[str | Path] = None):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set(title="Loss", xlabel="Epoch")
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set(title="Accuracy", xlabel="Epoch", ylim=(0, 1.02))
    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig
