"""Smoke tests that run on CPU without downloading pretrained weights."""

import torch
from PIL import Image
from torch import nn

from safety_detection import data, engine, model as model_lib
from safety_detection.predict import Predictor


def make_dataset(root, per_class=5):
    for i, name in enumerate(["Not Wearing Safety Gear", "Wearing Safety Gear"]):
        folder = root / name
        folder.mkdir(parents=True)
        for j in range(per_class):
            Image.new("RGB", (64, 48), (i * 200, j * 20, 50)).save(folder / f"{j}.jpg")


def test_stratified_split_is_disjoint_and_balanced():
    targets = [0] * 10 + [1] * 20
    train, val = data.stratified_split(targets, 0.3, seed=0)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(30))
    assert [targets[i] for i in val].count(0) == 3
    assert [targets[i] for i in val].count(1) == 6
    assert data.stratified_split(targets, 0.3, seed=0) == (train, val)


def test_train_save_and_predict(tmp_path):
    make_dataset(tmp_path / "ds")
    train_loader, val_loader, classes = data.create_dataloaders(
        tmp_path / "ds", batch_size=4, val_fraction=0.4, num_workers=0)
    assert classes == ["Not Wearing Safety Gear", "Wearing Safety Gear"]
    assert len(val_loader.dataset) == 4

    model = model_lib.build_model(len(classes), pretrained=False, freeze_backbone=True)
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["heads.weight", "heads.bias"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    ckpt = tmp_path / "ckpt.pt"
    history = engine.fit(model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
                         epochs=1, device=torch.device("cpu"), class_names=classes, checkpoint_path=ckpt)
    assert len(history["val_acc"]) == 1 and 0.0 <= history["val_acc"][0] <= 1.0

    probs = Predictor(ckpt, "cpu").predict(Image.new("RGB", (100, 80)))
    assert set(probs) == set(classes)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
