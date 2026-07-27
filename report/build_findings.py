"""Assemble a StructuredFindings object from the raw outputs of the
classifier + calibration + OOD stages -- the glue between the vision
pipeline and the LLM report layer.
"""
from __future__ import annotations

import numpy as np

from report.schema import Finding, StructuredFindings


def build_structured_findings(
    modality: str,
    labels: list[str],
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    ood_score: float,
    ood_threshold: float,
    heatmap_refs: list[str] | None = None,
    uncertain_band: tuple[float, float] = (0.4, 0.6),
) -> StructuredFindings:
    """`uncertain_band`: calibrated probabilities inside this range are flagged
    per-finding as uncertain (neither confidently positive nor negative)."""
    ood_flag = bool(ood_score < ood_threshold)

    findings = []
    for label, raw_p, cal_p in zip(labels, raw_probs, calibrated_probs):
        lo, hi = uncertain_band
        per_finding_uncertain = bool(lo <= cal_p <= hi) or ood_flag
        findings.append(
            Finding(
                label=label,
                probability=float(raw_p),
                calibrated_confidence=float(cal_p),
                flag_uncertain=per_finding_uncertain,
            )
        )

    return StructuredFindings(
        modality=modality,
        findings=findings,
        ood_flag=ood_flag,
        ood_score=float(ood_score),
        heatmap_refs=heatmap_refs or [],
    )
