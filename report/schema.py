"""Structured findings schema shared by the classifier pipeline and the LLM
report layer -- this is the contract both sides validate against, per the
plan's "define a strict JSON schema for findings" step.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Finding(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    flag_uncertain: bool


class StructuredFindings(BaseModel):
    modality: str  # "cxr" | "derm"
    findings: list[Finding]
    ood_flag: bool
    ood_score: float
    heatmap_refs: list[str] = Field(default_factory=list)

    def to_tool_schema_dict(self) -> dict:
        return self.model_dump()


# JSON schema handed to Claude as a tool definition (Claude API call #1), so
# the model's free-text-ish reasoning gets forced into this exact shape
# instead of being parsed out of prose.
FINDINGS_TOOL_SCHEMA = {
    "name": "emit_structured_findings",
    "description": "Emit the validated structured findings for this image, exactly matching the pipeline's schema.",
    "input_schema": {
        "type": "object",
        "properties": {
            "modality": {"type": "string", "enum": ["cxr", "derm"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "calibrated_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "flag_uncertain": {"type": "boolean"},
                    },
                    "required": ["label", "probability", "calibrated_confidence", "flag_uncertain"],
                },
            },
            "ood_flag": {"type": "boolean"},
            "ood_score": {"type": "number"},
            "heatmap_refs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["modality", "findings", "ood_flag", "ood_score"],
    },
}
