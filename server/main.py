"""FastAPI backend for the standalone web frontend (web/).

Wraps the exact same app/pipeline.py + app/load.py inference path the Gradio
demo uses -- one pipeline, two front doors. Run with:

    python -m server.main

or `uvicorn server.main:app --reload` during frontend development.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from app.load import load_modality_pipeline
from app.pipeline import log_inference, run_inference
from configs.config import load_config
from report.claude_report import DISCLAIMER_TEMPLATE

cfg = load_config()
PIPELINES = {
    "cxr": load_modality_pipeline("cxr", cfg),
    "derm": load_modality_pipeline("derm", cfg),
}
USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

app = FastAPI(title="Multimodal Medical Imaging Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FindingOut(BaseModel):
    label: str
    probability: float
    calibrated_confidence: float
    flag_uncertain: bool


class AnalyzeResponse(BaseModel):
    modality: str
    findings: list[FindingOut]
    ood_flag: bool
    ood_score: float
    report_text: str
    disclaimer: str
    overlay_image: str | None  # base64 PNG data URI
    overlay_label: str | None
    overlay_confident: bool
    used_llm: bool


def _array_to_data_uri(arr: np.ndarray) -> str:
    clipped = np.clip(arr, 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(clipped).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.get("/api/health")
def health():
    return {"status": "ok", "modalities": list(PIPELINES.keys()), "llm_enabled": USE_LLM}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (PNG/JPEG).")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    result = run_inference(image, PIPELINES, use_llm=USE_LLM)
    log_inference(cfg.app.audit_log_path, image, result)

    overlay_uri = _array_to_data_uri(result.overlays[0]) if result.overlays else None
    overlay_label = result.overlay_labels[0] if result.overlay_labels else None

    return AnalyzeResponse(
        modality=result.modality,
        findings=[FindingOut(**f.model_dump()) for f in result.findings.findings],
        ood_flag=result.findings.ood_flag,
        ood_score=result.findings.ood_score,
        report_text=result.report_text,
        disclaimer=DISCLAIMER_TEMPLATE,
        overlay_image=overlay_uri,
        overlay_label=overlay_label,
        overlay_confident=result.overlay_confident,
        used_llm=USE_LLM,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8000)))
