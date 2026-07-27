"""Grad-CAM explainability wrapper around `pytorch-grad-cam`.

One heatmap per predicted (above-threshold) label -- for CXR that means one
overlay per positive pathology, since a single chest X-ray can have several
simultaneous findings each localizing to a different region.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data.transforms import denormalize
from models.backbone import gradcam_target_layer


@dataclass
class HeatmapResult:
    label: str
    heatmap: np.ndarray          # raw CAM, HxW in [0, 1]
    overlay: np.ndarray          # heatmap blended onto the input image, HxWx3 uint8-range float


class GradCAMExplainer:
    """Reusable across both modalities -- this is the module the plan calls
    out as paying off once built modular in Week 3, then reused for
    dermoscopy in Week 4."""

    def __init__(self, model, target_layer=None, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        layer = target_layer or gradcam_target_layer(model)
        self.cam = GradCAM(model=model, target_layers=[layer])

    def explain(self, image_tensor: torch.Tensor, label_indices: list[int], label_names: list[str]) -> list[HeatmapResult]:
        """`image_tensor`: normalized, shape (1, 3, H, W)."""
        image_tensor = image_tensor.to(self.device)
        rgb_image = denormalize(image_tensor)[0].permute(1, 2, 0).cpu().numpy()

        results = []
        for idx in label_indices:
            targets = [ClassifierOutputTarget(idx)]
            grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)[0]  # (H, W) in [0,1]
            overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            results.append(
                HeatmapResult(label=label_names[idx], heatmap=grayscale_cam, overlay=overlay)
            )
        return results

    def explain_top_k(self, image_tensor: torch.Tensor, probs: np.ndarray, label_names: list[str], threshold: float = 0.5, top_k: int | None = None) -> list[HeatmapResult]:
        above = [i for i, p in enumerate(probs) if p >= threshold]
        if top_k is not None:
            above = sorted(above, key=lambda i: -probs[i])[:top_k]
        if not above:
            above = [int(np.argmax(probs))]  # always explain at least the top prediction
        return self.explain(image_tensor, above, label_names)


def save_overlay(result: HeatmapResult, path: str) -> None:
    Image.fromarray(result.overlay).save(path)
