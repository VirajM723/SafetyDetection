"""Inference on single images."""

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from .data import build_transforms
from .model import load_checkpoint


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    """Loads a trained checkpoint and returns class probabilities for images."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device | None = None):
        self.device = torch.device(device) if device else default_device()
        self.model, self.class_names = load_checkpoint(checkpoint_path, self.device)
        _, self.transform = build_transforms()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Dict[str, float]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1).squeeze(0).tolist()
        return dict(zip(self.class_names, probs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify images for safety gear.")
    parser.add_argument("images", nargs="+", type=Path, help="Image file(s) to classify")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vit_ppe.pt"))
    parser.add_argument("--device", default=None, help="cpu, cuda, ... (default: auto)")
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint, args.device)
    for path in args.images:
        start = time.perf_counter()
        probs = predictor.predict(Image.open(path))
        elapsed_ms = (time.perf_counter() - start) * 1000
        label = max(probs, key=probs.get)
        print(f"{path}: {label} ({probs[label]:.1%}) [{elapsed_ms:.0f} ms]")


if __name__ == "__main__":
    main()
