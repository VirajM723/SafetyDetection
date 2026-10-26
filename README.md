# Safety Gear Detection with Vision Transformers

Classifies whether a worker in an image is wearing personal protective equipment (PPE), such as a hard hat. It fine-tunes a **ViT-B/16** Vision Transformer, pretrained on ImageNet, from TorchVision. The repo includes a training CLI, a single-image inference CLI and a **Gradio** web demo.

> **Scope:** this is **image-level binary classification** (*wearing* / *not wearing* safety gear). It does not localise people or individual PPE items. See [Roadmap](#roadmap).

---

## Results

On the original run in Google Colab (T4 GPU) the model was trained for 10 epochs with a frozen backbone, so only the classification head learned. It reached:

| Split | Images | Accuracy (epoch 10) |
|-------|-------:|--------------------:|
| Train | 248 | 98.4 % |
| Held-out test | 107 | **96.1 %** |

Validation accuracy was still rising at epoch 10. The held-out set is small (107 images), so treat these numbers as indicative only. With n = 107, ±1 image moves accuracy by about 1 percentage point. Re-run `python -m safety_detection.train` to get numbers for your own split. Training also writes `checkpoints/history.json` and `checkpoints/training_curves.png`.

---

## How it works

| Stage | Details |
|-------|---------|
| **Data** | *Wearing Safety Gear or Not* dataset (Kaggle): 355 images in 2 classes, read with `torchvision.datasets.ImageFolder` |
| **Split** | Stratified and seeded 70 / 30 train / validation split, so it is reproducible and class-balanced |
| **Preprocessing** | Resize to 224×224, then normalise with ImageNet mean/std to match the pretrained weights |
| **Augmentation** | Training split only: `RandomResizedCrop`, horizontal flip, colour jitter |
| **Model** | `vit_b_16` (~86 M params) with ImageNet weights. The backbone is frozen and a new `768 → 2` linear head is trained (about 1.5 k trainable params). `--unfreeze` fine-tunes the whole network instead |
| **Training** | Cross-entropy loss, AdamW optimiser. The checkpoint with the best validation accuracy is saved |
| **Evaluation** | Accuracy per sample (not averaged per batch) plus a confusion matrix |

---

## Project structure

```
SafetyDetection/
├── safety_detection/          # Python package
│   ├── data.py                # transforms, stratified split, dataloaders
│   ├── model.py               # ViT construction + checkpoint save/load
│   ├── engine.py              # train / evaluate loops, training curves
│   ├── train.py               # CLI: python -m safety_detection.train
│   ├── predict.py             # Predictor class + CLI: python -m safety_detection.predict
│   └── app.py                 # Gradio demo: python -m safety_detection.app
├── tests/test_pipeline.py     # CPU smoke tests (no weight download needed)
├── safety_equipment.ipynb     # notebook walkthrough built on the package
├── safety-detection-ppt.pptx  # project presentation
├── safety-equipment-paper.pdf # reference: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
└── requirements.txt
```

---

## Getting started

### 1. Install

Requires Python 3.10+.

```bash
git clone https://github.com/VirajM723/SafetyDetection.git
cd SafetyDetection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download the *Wearing Safety Gear or Not* dataset from Kaggle. Place it in the repo root with one folder per class:

```
GearedorNot/
├── Not Wearing Safety Gear/
│   └── *.jpg
└── Wearing Safety Gear/
    └── *.jpg
```

Any ImageFolder-style directory works, including directories with more than two classes.

### 3. Train

```bash
python -m safety_detection.train --data-dir GearedorNot --epochs 10
```

Useful flags: `--lr`, `--batch-size`, `--val-fraction`, `--seed`, `--output-dir`, and `--unfreeze` (full fine-tuning; use a smaller LR such as `--lr 1e-5`). Run with `--help` for the full list.

### 4. Predict

```bash
python -m safety_detection.predict path/to/image1.jpg path/to/image2.png
# path/to/image1.jpg: Wearing Safety Gear (97.3%) [85 ms]
```

From Python:

```python
from PIL import Image
from safety_detection.predict import Predictor

predictor = Predictor("checkpoints/vit_ppe.pt")
predictor.predict(Image.open("worker.jpg"))
# {'Not Wearing Safety Gear': 0.03, 'Wearing Safety Gear': 0.97}
```

### 5. Run the web demo

```bash
python -m safety_detection.app            # add --share for a public link
```

### 6. Run tests

```bash
pip install pytest
pytest
```

The tests build a tiny synthetic dataset and run on CPU without downloading pretrained weights.

---

## Tech stack

Python · PyTorch · TorchVision (ViT-B/16) · Gradio · Matplotlib

---

## Roadmap

- Move from classification to **object detection** (e.g. YOLO / DETR), so that multiple workers and individual items (helmet, vest, goggles, gloves) are localised.
- Train on a larger and more varied dataset, and report results with k-fold cross-validation.
- **Real-time video** inference from webcam / CCTV streams, with alerts.
- Export to ONNX / TorchScript for edge deployment.

---

## Author

**Viraj Mankani**
