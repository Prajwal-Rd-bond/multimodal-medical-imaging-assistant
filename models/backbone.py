"""Classifier backbones for both modalities.

Both CXR (multi-label, sigmoid) and dermoscopy (multi-class, softmax) reuse
the same timm backbone factory -- only the head and loss differ, which is
handled by the training engine, not here.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """timm backbone + linear head, with optional early-stage freezing.

    `freeze_stages` freezes the first N of timm's `feature_info` stages --
    this is what the plan means by "frozen early layers to cut compute": the
    backbone's early convolutional stages learn generic edge/texture filters
    that transfer as-is, so only the later stages + head need gradients.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_stages: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes)

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, n: int) -> None:
        # DenseNet/VGG-style timm models bundle every conv stage into one
        # top-level `features` Sequential (stem, denseblock1, transition1, ...,
        # denseblock4, norm5) -- freezing top-level children of the backbone
        # itself would freeze that whole Sequential as a single "stage" (i.e.
        # the entire feature extractor) for any n >= 1. Walk one level deeper
        # into `features` when present so `n` actually selects individual
        # stages; fall back to top-level children (ResNet/EfficientNet-style,
        # where stages already are top-level modules) otherwise.
        container = getattr(self.backbone, "features", self.backbone)
        modules = dict(container.named_children())
        for i, (_, module) in enumerate(modules.items()):
            if i >= n:
                break
            for p in module.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer features -- used by Grad-CAM target layers and
        the Mahalanobis OOD detector."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(self.dropout(feats))


def build_classifier(
    modality: str,
    num_classes: int,
    backbone_name: str = "densenet121",
    pretrained: bool = True,
    freeze_stages: int = 0,
    dropout: float = 0.2,
) -> ClassifierHead:
    if modality not in ("cxr", "derm"):
        raise ValueError(f"Unknown modality {modality!r}, expected 'cxr' or 'derm'")
    return ClassifierHead(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_stages=freeze_stages,
        dropout=dropout,
    )


def gradcam_target_layer(model: ClassifierHead):
    """Best-effort default Grad-CAM target layer for common timm CNN backbones."""
    backbone = model.backbone
    name = backbone.__class__.__name__.lower()
    if "densenet" in name:
        return backbone.features.norm5 if hasattr(backbone.features, "norm5") else list(backbone.features.children())[-1]
    if "resnet" in name:
        return backbone.layer4[-1]
    if "efficientnet" in name:
        return backbone.conv_head if hasattr(backbone, "conv_head") else backbone.blocks[-1]
    # Generic fallback: last conv-ish module found by walking children.
    last = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError(f"Could not infer a Grad-CAM target layer for backbone {name}")
    return last
