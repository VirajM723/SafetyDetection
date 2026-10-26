"""Gradio demo: ``python -m safety_detection.app --checkpoint checkpoints/vit_ppe.pt``."""

import argparse
from pathlib import Path

from .predict import Predictor


def build_interface(predictor: Predictor):
    import gradio as gr

    return gr.Interface(
        fn=predictor.predict,
        inputs=gr.Image(type="pil", label="Image"),
        outputs=gr.Label(num_top_classes=len(predictor.class_names), label="Prediction"),
        title="Safety Gear Identification",
        description="Upload a photo of a worker to check whether they are wearing safety gear.",
        flagging_mode="never",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the safety gear demo app.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vit_ppe.pt"))
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()
    build_interface(Predictor(args.checkpoint)).launch(share=args.share)


if __name__ == "__main__":
    main()
