"""End-to-end inference pipeline: image in -> (overlay, prose report, findings) out.

This is the single object both the Gradio app and any batch/eval script
should call -- keeps the "one modular pipeline reused everywhere" property
the plan asks for.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate.temperature_scaling import TemperatureScaler
from data.transforms import eval_transforms
from explain.gradcam import GradCAMExplainer
from models.modality_router import detect_modality
from ood.detectors import MahalanobisDetector, entropy_score
from report.build_findings import build_structured_findings
from report.claude_report import DISCLAIMER_TEMPLATE, generate_prose_report
from report.schema import StructuredFindings


@dataclass
class ModalityPipeline:
    """One modality's worth of loaded artifacts (model + calibrator + labels
    + explainer), assembled once at app startup."""

    modality: str
    model: torch.nn.Module
    label_names: list[str]
    temperature_scaler: TemperatureScaler
    explainer: GradCAMExplainer
    task: str  # "multilabel" | "multiclass"
    ood_threshold: float
    ood_method: str = "entropy"  # "entropy" | "mahalanobis"
    mahalanobis: MahalanobisDetector | None = None
    image_size: int = 224
    device: torch.device = torch.device("cpu")


@dataclass
class InferenceResult:
    modality: str
    overlays: list[np.ndarray]
    overlay_labels: list[str]
    findings: StructuredFindings
    report_text: str


def _probs_from_logits(logits: torch.Tensor, task: str) -> np.ndarray:
    if task == "multilabel":
        return torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()


def run_inference(
    image: Image.Image,
    pipelines: dict[str, ModalityPipeline],
    use_llm: bool = True,
    disclaimer: str = DISCLAIMER_TEMPLATE,
) -> InferenceResult:
    decision = detect_modality(image)
    pipeline = pipelines[decision.modality]

    transform = eval_transforms(pipeline.image_size)
    image_tensor = transform(image).unsqueeze(0).to(pipeline.device)

    with torch.no_grad():
        logits = pipeline.model(image_tensor)
        raw_probs = _probs_from_logits(logits, pipeline.task)
        calibrated_logits = pipeline.temperature_scaler(logits.cpu())
        calibrated_probs = _probs_from_logits(calibrated_logits, pipeline.task)

    multilabel = pipeline.task == "multilabel"
    if pipeline.ood_method == "mahalanobis" and pipeline.mahalanobis is not None:
        with torch.no_grad():
            features = pipeline.model.forward_features(image_tensor).cpu().numpy()
        ood_score = float(pipeline.mahalanobis.score(features)[0])
    else:
        ood_score = float(entropy_score(raw_probs[None, :], multilabel=multilabel)[0])

    heatmaps = pipeline.explainer.explain_top_k(image_tensor, raw_probs, pipeline.label_names, threshold=0.5, top_k=3)

    findings = build_structured_findings(
        modality=pipeline.modality,
        labels=pipeline.label_names,
        raw_probs=raw_probs,
        calibrated_probs=calibrated_probs,
        ood_score=ood_score,
        ood_threshold=pipeline.ood_threshold,
        heatmap_refs=[h.label for h in heatmaps],
    )

    if use_llm:
        report_text = generate_prose_report(findings, disclaimer=disclaimer)
    else:
        report_text = _fallback_report(findings, disclaimer)

    return InferenceResult(
        modality=pipeline.modality,
        overlays=[h.overlay for h in heatmaps],
        overlay_labels=[h.label for h in heatmaps],
        findings=findings,
        report_text=report_text,
    )


def _fallback_report(findings: StructuredFindings, disclaimer: str) -> str:
    """No-LLM fallback (no ANTHROPIC_API_KEY set) so the demo still runs end
    to end without a Claude API key -- clearly worse prose, same guardrails."""
    lines = [disclaimer, ""]
    if findings.ood_flag:
        lines.append("This input looks unusual relative to the training distribution; treat every finding below as inconclusive.")
    for f in findings.findings:
        status = "uncertain" if f.flag_uncertain else "flagged"
        lines.append(f"- {f.label}: {status} (calibrated confidence {f.calibrated_confidence:.0%})")
    return "\n".join(lines)


def log_inference(audit_log_path: str | Path, image: Image.Image, result: InferenceResult) -> None:
    """Append-only audit trail: input hash, prediction, confidence, flags --
    per the plan's guardrails checklist, and a stand-in for what a real
    deployment would log at production scale (with proper access controls)."""
    path = Path(audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_bytes = image.tobytes()
    input_hash = hashlib.sha256(image_bytes).hexdigest()

    record = {
        "timestamp": time.time(),
        "input_hash": input_hash,
        "modality": result.modality,
        "findings": [f.model_dump() for f in result.findings.findings],
        "ood_flag": result.findings.ood_flag,
        "ood_score": result.findings.ood_score,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
