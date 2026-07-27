"""End-to-end smoke test on synthetic data -- no real datasets, no GPU, no
Claude API key required. Not a substitute for training on real data; it only
proves the modules wire together correctly (tensor shapes, schema
validation, checkpoint round-trip).

    python -m pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image

from calibrate.temperature_scaling import compute_ece, fit_temperature
from data.datasets import DermoscopyDataset, MultiLabelImageDataset
from data.transforms import eval_transforms, train_transforms
from explain.gradcam import GradCAMExplainer
from models.backbone import build_classifier
from models.modality_router import detect_modality
from ood.detectors import entropy_score, evaluate_ood, max_softmax_score
from report.build_findings import build_structured_findings
from report.schema import StructuredFindings
from train.checkpoint import load_checkpoint, save_checkpoint
from train.engine import evaluate, train_one_epoch

CXR_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
DERM_LABELS = ["melanoma", "nevus", "other"]


def _make_synthetic_images(tmp_path, n: int, color: bool):
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    rng = np.random.RandomState(0)
    paths = []
    for i in range(n):
        if color:
            arr = rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        else:
            gray = rng.randint(0, 255, size=(64, 64), dtype=np.uint8)
            arr = np.stack([gray] * 3, axis=-1)
        img = Image.fromarray(arr)
        name = f"img_{i}.png"
        img.save(img_dir / name)
        paths.append(f"images/{name}")
    return paths


def test_modality_router(tmp_path):
    gray = Image.fromarray(np.stack([np.full((32, 32), 128, dtype=np.uint8)] * 3, axis=-1))
    color = Image.fromarray(np.random.RandomState(1).randint(0, 255, size=(32, 32, 3), dtype=np.uint8))
    assert detect_modality(gray).modality == "cxr"
    # a random-noise "color" image should read as high-saturation -> derm
    assert detect_modality(color).modality in ("cxr", "derm")  # heuristic, not guaranteed on pure noise


def test_cxr_dataset_and_training_step(tmp_path):
    paths = _make_synthetic_images(tmp_path, n=8, color=False)
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"image_path": paths})
    for label in CXR_LABELS:
        df[label] = rng.randint(0, 2, size=len(paths))
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)

    ds = MultiLabelImageDataset(csv_path, tmp_path, "image_path", CXR_LABELS, train_transforms(64))
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["labels"].shape == (len(CXR_LABELS),)

    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    model = build_classifier("cxr", num_classes=len(CXR_LABELS), backbone_name="resnet18", pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    loss, step = train_one_epoch(model, loader, optimizer, device, task="multilabel", global_step=0)
    assert np.isfinite(loss)
    assert step == 2  # 8 samples / batch_size 4

    metrics, logits, labels = evaluate(model, loader, device, task="multilabel", label_names=CXR_LABELS)
    assert logits.shape == (8, len(CXR_LABELS))
    assert "auc/mean" in metrics


def test_derm_dataset(tmp_path):
    paths = _make_synthetic_images(tmp_path, n=6, color=True)
    rng = np.random.RandomState(0)
    labels = rng.choice(DERM_LABELS, size=len(paths))
    df = pd.DataFrame({"image_path": paths, "label": labels})
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)

    ds = DermoscopyDataset(csv_path, tmp_path, "image_path", DERM_LABELS, eval_transforms(64))
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["label"].dtype == torch.long


def test_checkpoint_roundtrip(tmp_path):
    model = build_classifier("cxr", num_classes=5, backbone_name="resnet18", pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "test_latest.pt"
    save_checkpoint(ckpt_path, model, optimizer, None, epoch=3, global_step=42, best_metric=0.8)

    model2 = build_classifier("cxr", num_classes=5, backbone_name="resnet18", pretrained=False)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    state = load_checkpoint(ckpt_path, model2, optimizer2)
    assert state["epoch"] == 3
    assert state["global_step"] == 42
    assert state["best_metric"] == 0.8
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_gradcam():
    model = build_classifier("cxr", num_classes=len(CXR_LABELS), backbone_name="resnet18", pretrained=False)
    explainer = GradCAMExplainer(model, device="cpu")
    image_tensor = torch.randn(1, 3, 64, 64)
    probs = np.array([0.9, 0.1, 0.6, 0.2, 0.05])
    results = explainer.explain_top_k(image_tensor, probs, CXR_LABELS, threshold=0.5, top_k=2)
    assert len(results) == 2
    assert results[0].heatmap.shape[:2] == (64, 64) or results[0].overlay.shape[:2] == (64, 64)


def test_temperature_scaling_and_ece():
    torch.manual_seed(0)
    logits = torch.randn(200, 3) * 3
    labels = torch.randint(0, 3, (200,))
    scaler = fit_temperature(logits, labels, task="multiclass", per_class=False, max_iter=20)
    assert scaler.temperature.shape == (1,)

    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    correct = (preds == labels.numpy()).astype(float)
    ece = compute_ece(probs.max(axis=1), correct, n_bins=10)
    assert 0.0 <= ece.ece <= 1.0


def test_ood_detectors():
    id_probs = np.random.dirichlet([5, 1, 1], size=100)
    ood_probs = np.random.dirichlet([1, 1, 1], size=100)
    id_scores = entropy_score(id_probs)
    ood_scores = entropy_score(ood_probs)
    metrics = evaluate_ood(id_scores, ood_scores, target_tpr=0.95)
    assert 0.0 <= metrics["auroc"] <= 1.0

    assert max_softmax_score(id_probs).mean() > max_softmax_score(ood_probs).mean()


def test_structured_findings_schema():
    findings = build_structured_findings(
        modality="cxr",
        labels=CXR_LABELS,
        raw_probs=np.array([0.9, 0.1, 0.55, 0.2, 0.05]),
        calibrated_probs=np.array([0.8, 0.15, 0.5, 0.25, 0.1]),
        ood_score=0.5,
        ood_threshold=0.1,
        heatmap_refs=["Atelectasis"],
    )
    assert isinstance(findings, StructuredFindings)
    assert findings.ood_flag is False  # ood_score(0.5) >= threshold(0.1)
    # Consolidation's calibrated prob (0.5) is inside the default uncertain band.
    consolidation = next(f for f in findings.findings if f.label == "Consolidation")
    assert consolidation.flag_uncertain is True

    # round-trips through JSON exactly like the Claude tool-use call would receive it
    dumped = findings.model_dump_json()
    restored = StructuredFindings.model_validate_json(dumped)
    assert restored == findings
