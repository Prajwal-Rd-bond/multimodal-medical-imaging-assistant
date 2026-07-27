"""Gradio demo: upload -> modality auto-detect -> classifier -> Grad-CAM
overlay -> calibration -> OOD flag -> Claude prose report.

    python -m app.gradio_app

Set ANTHROPIC_API_KEY to get real Claude reports; without it, the app falls
back to a templated (non-LLM) report so the demo still runs end to end.
"""
from __future__ import annotations

import os

import gradio as gr
from PIL import Image

from app.load import load_modality_pipeline
from app.pipeline import log_inference, run_inference
from configs.config import load_config

cfg = load_config()
PIPELINES = {
    "cxr": load_modality_pipeline("cxr", cfg),
    "derm": load_modality_pipeline("derm", cfg),
}
USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Persistent, unavoidable disclaimer -- shown regardless of what the model
# says, per the plan's responsible-AI guardrails checklist.
DISCLAIMER_MD = (
    "> **Research demo, not a diagnostic device.** Not reviewed or approved for "
    "clinical use. Do not use this output to make or inform any medical decision. "
    "Always consult a qualified healthcare professional."
)


def analyze(image: Image.Image):
    if image is None:
        return None, None, "Upload an image first.", "—"

    result = run_inference(image, PIPELINES, use_llm=USE_LLM)
    log_inference(cfg.app.audit_log_path, image, result)

    overlay = result.overlays[0] if result.overlays else None
    overlay_caption = result.overlay_labels[0] if result.overlay_labels else ""

    flags = []
    if result.findings.ood_flag:
        flags.append("⚠️ Out-of-distribution input -- treat all findings as unreliable")
    for f in result.findings.findings:
        if f.flag_uncertain:
            flags.append(f"⚠️ {f.label}: uncertain")
    flags_text = "\n".join(flags) if flags else "No uncertainty flags raised."

    detected = f"Detected modality: **{result.modality.upper()}**"
    return overlay, overlay_caption, result.report_text, f"{detected}\n\n{flags_text}"


with gr.Blocks(title="Multimodal Medical Imaging Assistant") as demo:
    gr.Markdown("# Multimodal Medical Imaging Assistant")
    gr.Markdown(DISCLAIMER_MD)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a chest X-ray or dermoscopy image")
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            overlay_output = gr.Image(label="Grad-CAM overlay (top finding)")
            overlay_caption = gr.Textbox(label="Overlay label", interactive=False)

    with gr.Row():
        status_output = gr.Markdown(label="Modality / flags")
        report_output = gr.Textbox(label="Structured report", lines=12, interactive=False)

    submit_btn.click(
        fn=analyze,
        inputs=[image_input],
        outputs=[overlay_output, overlay_caption, report_output, status_output],
    )

    gr.Markdown(DISCLAIMER_MD)


if __name__ == "__main__":
    demo.launch(server_name=cfg.app.host, server_port=cfg.app.port)
