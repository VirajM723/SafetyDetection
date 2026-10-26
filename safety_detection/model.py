"""Model construction and checkpoint I/O."""

from pathlib import Path
from typing import List, Tuple

import torch
import torchvision
from torch import nn


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """ViT-B/16 with a new linear classification head.

    With ``freeze_backbone`` only the head is trained (linear probing), which
    is fast and works well on small datasets.
    """
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = torchvision.models.vit_b_16(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.heads = nn.Linear(model.hidden_dim, num_classes)
    return model


def save_checkpoint(path: str | Path, model: nn.Module, class_names: List[str], **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "class_names": list(class_names), **metadata}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Tuple[nn.Module, List[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), class_names
