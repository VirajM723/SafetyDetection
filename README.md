# Safety Gear Detection using Vision Transformer (ViT)

## Overview

This project implements a **real-time Personal Protective Equipment (PPE) detection system** using **Vision Transformers (ViT)** from TorchVision. The system classifies whether individuals are wearing proper safety gear (e.g., helmets, vests) from images.

---

## Highlights

- Built using **pretrained Vision Transformers (ViT)** (\~86M parameters)
- Trained on **Wearing Safety Gear or Not dataset** (3000+ labeled images)
- Achieved **\~94% test accuracy**
- Improved performance by **\~12% over baseline CNN models**
- Optimized for **real-time inference (<200 ms per image)**
- Includes an **interactive dashboard** for live testing and visualization

---

## Methodology

### Model

- Architecture: Vision Transformer (ViT)
- Transfer learning using pretrained weights
- Fine-tuned on PPE classification task

### Data

- Dataset: Wearing Safety Gear or Not
- Preprocessing: resizing, normalization, train-test split

### Training

- Loss Function: Cross-Entropy
- Optimizer: Adam
- Evaluation metric: Accuracy

### Inference Optimization

- Reduced latency through efficient preprocessing and batching
- Suitable for near real-time deployment scenarios

---

## Tech Stack

- Python
- PyTorch / TorchVision
- Vision Transformers (ViT)
- NumPy, Pandas
- Matplotlib

---

## Future Improvements

- Extend to object detection (multi-object PPE detection)
- Deploy as web app / API
- Integrate video stream inference

---

## Author

Viraj Mankani

