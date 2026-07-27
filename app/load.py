"""Load trained checkpoints + fitted calibrators into the ModalityPipeline
objects the app/eval code runs inference through.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.pipeline import ModalityPipeline
from calibrate.temperature_scaling import TemperatureScaler
from explain.gradcam import GradCAMExplainer
from models.backbone import build_classifier
from ood.detectors import MahalanobisDetector
from train.device import get_device


def load_modality_pipeline(
    modality: str,
    cfg,
    checkpoint_path: str | Path | None = None,
    temperature_path: str | Path | None = None,
) -> ModalityPipeline:
    device = get_device()
    task = "multilabel" if modality == "cxr" else "multiclass"
    dcfg = cfg.data.cxr if modality == "cxr" else cfg.data.derm
    label_names = list(dcfg.label_cols)

    model = build_classifier(
        modality=modality,
        num_classes=len(label_names),
        backbone_name=cfg.model.backbone,
        pretrained=False,  # weights come from the checkpoint, not ImageNet, at inference time
        freeze_stages=0,
        dropout=cfg.model.dropout,
    ).to(device)

    ckpt_path = Path(checkpoint_path or Path(cfg.train.checkpoint_dir) / f"{modality}_best.pt")
    if ckpt_path.exists():
        # weights_only=False: these are checkpoints this repo trained and saved itself, not
        # third-party files -- and the optimizer/scheduler state includes plain Python floats
        # that trip torch 2.6+'s stricter weights_only=True default.
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
    else:
        print(f"WARNING: no checkpoint found at {ckpt_path} -- using untrained/ImageNet weights. "
              f"Train first with `python -m train.train --modality {modality}`.")
    model.eval()

    temperature_scaler = TemperatureScaler(len(label_names), per_class=(task == "multilabel"))
    temp_path = Path(temperature_path or Path(cfg.train.checkpoint_dir) / f"{modality}_temperature.pt")
    if temp_path.exists():
        temperature_scaler.load_state_dict(torch.load(temp_path, map_location="cpu"))
    else:
        print(f"No fitted temperature at {temp_path} -- calibrated_confidence will equal raw probability "
              f"until you run `python -m calibrate.fit_temperature --modality {modality}`.")

    explainer = GradCAMExplainer(model, device=device)

    ood_method = cfg.ood.method
    ood_threshold_path = Path(cfg.train.checkpoint_dir) / f"{modality}_ood_threshold.json"
    if ood_threshold_path.exists():
        thresholds = json.loads(ood_threshold_path.read_text())["thresholds"]
        ood_threshold = thresholds.get(ood_method)
        if ood_threshold is None:
            print(f"No fitted '{ood_method}' threshold in {ood_threshold_path} (has {list(thresholds)}) -- "
                  f"falling back to untuned placeholder (-1.0).")
            ood_threshold = -1.0
    else:
        ood_threshold = -1.0  # untuned placeholder; run `python -m eval.run_eval --modality {modality}` to fit a real one
        print(f"No tuned OOD threshold at {ood_threshold_path} -- using untuned placeholder ({ood_threshold}). "
              f"Run `python -m eval.run_eval --modality {modality}` to fit one from real validation scores.")

    mahalanobis = None
    if ood_method == "mahalanobis":
        maha_path = Path(cfg.train.checkpoint_dir) / f"{modality}_mahalanobis.npz"
        if maha_path.exists():
            data = np.load(maha_path)
            mahalanobis = MahalanobisDetector(means=data["means"], precision=data["precision"])
        else:
            print(f"ood.method is 'mahalanobis' but no fitted detector at {maha_path} -- "
                  f"falling back to entropy scoring. Run `python -m eval.run_eval --modality {modality} "
                  f"--ood-dataset <dir>` to fit one.")
            ood_method = "entropy"

    return ModalityPipeline(
        modality=modality,
        model=model,
        label_names=label_names,
        temperature_scaler=temperature_scaler,
        explainer=explainer,
        task=task,
        ood_threshold=ood_threshold,
        ood_method=ood_method,
        mahalanobis=mahalanobis,
        image_size=dcfg.image_size,
        device=device,
    )
