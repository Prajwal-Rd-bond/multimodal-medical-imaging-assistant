"""Modality auto-detection: is this upload a chest X-ray or a dermoscopy image?

Per the plan this only needs to be a "simple heuristic or lightweight
classifier" -- CXRs and dermoscopy images differ enormously in low-level
statistics (CXR: near-grayscale, low saturation, high contrast, roughly
square lung-field composition; dermoscopy: strongly colored skin tones, often
circular vignette from the dermatoscope), so a cheap heuristic on color
statistics is enough and needs no training data of its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class ModalityDecision:
    modality: str  # "cxr" | "derm"
    confidence: float
    saturation_mean: float


def detect_modality(image: Image.Image, saturation_threshold: float = 0.12) -> ModalityDecision:
    """Cheap, dependency-free heuristic based on HSV saturation.

    Chest X-rays are acquired and stored as grayscale (even when saved as
    3-channel PNG/JPEG, R≈G≈B per pixel), so mean saturation is near zero.
    Dermoscopy images are color clinical photographs with real chroma.
    """
    hsv = np.array(image.convert("HSV"), dtype=np.float32)
    saturation = hsv[..., 1] / 255.0
    sat_mean = float(saturation.mean())

    if sat_mean < saturation_threshold:
        modality = "cxr"
        # Further from threshold = more confident either direction.
        confidence = 1.0 - (sat_mean / saturation_threshold)
    else:
        modality = "derm"
        confidence = min(1.0, (sat_mean - saturation_threshold) / saturation_threshold)

    return ModalityDecision(modality=modality, confidence=round(float(confidence), 3), saturation_mean=round(sat_mean, 4))
