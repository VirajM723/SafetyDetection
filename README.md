# Safety Gear Classification with a Vision Transformer

Classifies whether a worker in an image is wearing personal protective equipment (PPE), such as a hard hat. It fine-tunes the classification head of a **ViT-B/16** Vision Transformer (pretrained on ImageNet, from TorchVision) on a small Kaggle dataset. The repo includes a training CLI, a single-image inference CLI and a **Gradio** web demo.

> **Scope:** this is **image-level binary classification** (*wearing* / *not wearing* safety gear). It gives one label per image. It does **not** localise people or individual PPE items, and it does not process video. It is a student project, not a production safety system. See [Limitations](#limitations) and [Roadmap](#roadmap).

---

## Results

One training run in Google Colab (T4 GPU): 10 epochs, frozen backbone, so only the classification head learned.

| Split | Images | Accuracy (epoch 10) |
|-------|-------:|--------------------:|
| Train | 248 | 98.4 % |
| Validation | 107 | 96.1 % |

Read these numbers with care:

- **The validation split is not an independent test set.** The same 107 images are used to choose which epoch's checkpoint to keep *and* to report accuracy, so 96.1 % is somewhat optimistic.
- **The sample is small.** 96.1 % is about 103 / 107 correct, and one image moves accuracy by about 1 percentage point. A 95 % confidence interval is roughly 91 – 98.5 %.
- **It comes from a single split and seed.** No cross-validation or repeated runs were done.
- **No checkpoint or training logs are committed**, so the numbers above can't be checked from this repo alone. Re-run `python -m safety_detection.train` to get numbers for your own split. Training writes `checkpoints/history.json` and `checkpoints/training_curves.png`.
- Validation accuracy was still rising at epoch 10, so longer training may help.
- No comparison against a CNN baseline (e.g. ResNet-50) has been run, so this repo does not show that ViT beats simpler models on this task.

---

## How it works

| Stage | Details |
|-------|---------|
| **Data** | *Wearing Safety Gear or Not* dataset (Kaggle): 355 images in 2 classes, read with `torchvision.datasets.ImageFolder` |
| **Split** | Stratified, seeded 70 / 30 train / validation split (reproducible and class-balanced). There is no separate test split |
| **Preprocessing** | Resize to 224×224 (non-square images are stretched, not cropped), then normalise with ImageNet mean/std to match the pretrained weights |
| **Augmentation** | Training split only: `RandomResizedCrop`, horizontal flip, colour jitter |
| **Model** | `vit_b_16` (~86 M params) with ImageNet weights. The backbone is frozen and a new `768 → 2` linear head is trained (1,538 trainable params). `--unfreeze` fine-tunes the whole network instead (not benchmarked here) |
| **Training** | Cross-entropy loss, AdamW optimiser (lr 1e-3, weight decay 1e-4), no LR schedule or early stopping. The checkpoint with the best validation accuracy is saved |
| **Evaluation** | Per-sample accuracy. `engine.evaluate` also returns a confusion matrix (printed in the notebook, not by the training CLI). Precision / recall per class are not reported |

---

## Limitations

- **Classification, not detection.** The model can't say *which* person is missing gear, and images with several workers get a single label.
- **Small, single-source dataset.** 355 web images. Real CCTV footage (high angles, low light, motion blur, small figures) looks different, so expect accuracy to drop on real site cameras without more data.
- **Mainly hard hats.** The classes are just "wearing" / "not wearing" safety gear. Individual items (vests, goggles, gloves) are not distinguished.
- **No real-time or video pipeline.** Inference is one image at a time from the CLI, Python or the Gradio demo. Webcam/CCTV streaming and alerting are not implemented.
- **Accuracy is the only reported metric.** For safety use, recall on the *Not Wearing* class (missed violations) matters more than overall accuracy, and it is not measured or tuned.
- **Confidence scores aren't calibrated.** A "97 %" output is the softmax score, not a verified probability.
- **The Gradio app is a demo**, not a deployment: no authentication, batching or monitoring.

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
├── safety-detection-ppt.pptx  # project presentation (describes the longer-term vision)
├── safety-equipment-paper.pdf # reference: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
└── requirements.txt
```

---

## Getting started

### 1. Install

Requires Python 3.10+. A GPU is recommended for training. CPU works, but slowly, since every image still passes through the full ViT.

```bash
git clone https://github.com/VirajM723/SafetyDetection.git
cd SafetyDetection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the pretrained ViT-B/16 weights (~330 MB) from TorchVision.

### 2. Get the dataset

The dataset is not included. Download *Wearing Safety Gear or Not* from Kaggle and place it in the repo root with one folder per class:

```
GearedorNot/
├── Not Wearing Safety Gear/
│   └── *.jpg
└── Wearing Safety Gear/
    └── *.jpg
```

Any ImageFolder-style directory works, including ones with more than two classes.

### 3. Train

```bash
python -m safety_detection.train --data-dir GearedorNot --epochs 10
```

Useful flags: `--lr`, `--batch-size`, `--val-fraction`, `--seed`, `--output-dir`, `--num-workers` (set `0` if worker processes cause problems on Windows), and `--unfreeze` (full fine-tuning; use a smaller LR such as `--lr 1e-5`). Run with `--help` for the full list.

### 4. Predict

```bash
python -m safety_detection.predict path/to/image1.jpg path/to/image2.png
```

It prints one line per image: the predicted class, its softmax score and the inference time. Latency depends on your hardware.

From Python:

```python
from PIL import Image
from safety_detection.predict import Predictor

predictor = Predictor("checkpoints/vit_ppe.pt")
predictor.predict(Image.open("worker.jpg"))
# returns {class_name: probability}, e.g. {'Not Wearing Safety Gear': ..., 'Wearing Safety Gear': ...}
```

### 5. Run the web demo

```bash
python -m safety_detection.app            # add --share for a temporary public link
```

### 6. Run tests

```bash
pip install pytest
pytest
```

The tests build a tiny synthetic dataset and run on CPU without downloading pretrained weights. They check that the pipeline runs end to end (split → train → save → load → predict). They do **not** measure model accuracy.

---

## Tech stack

Python · PyTorch · TorchVision (ViT-B/16) · Gradio · Matplotlib · pytest

---

## Roadmap

Roughly in priority order:

1. Add a separate test split (or k-fold cross-validation) and report per-class precision / recall alongside accuracy.
2. Compare against CNN baselines and partial fine-tuning (unfreezing the last few ViT blocks).
3. Train on a larger, more varied dataset, ideally including real site-camera footage.
4. Move from classification to **object detection** (e.g. YOLO / DETR), so that multiple workers and individual items (helmet, vest, goggles, gloves) are localised.
5. **Video** inference from webcam / CCTV streams, with temporal smoothing and alerts.
6. Export to ONNX / TorchScript for edge deployment.

---

## Author

**Viraj Mankani**
