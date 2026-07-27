---
title: Multimodal Medical Imaging Assistant
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.20.0"
app_file: app.py
pinned: false
license: mit
---

# Multimodal Medical Imaging Assistant

Chest X-ray + dermoscopy classification, Grad-CAM explainability, calibration
and OOD detection, LLM-generated structured reports, deployed demo. A
portfolio project demonstrating a *responsible* imaging pipeline, not a
diagnostic tool -- see [Guardrails](#responsible-ai--guardrails) below.

> **This is a research demo, not a diagnostic device.** Every prediction it
> makes is disclaimed as such, in the UI and in every generated report.

## Architecture

```
Image upload (CXR or dermoscopy)
   -> Preprocessing (resize, normalize, modality auto-detect)
   -> Classifier backbone (per modality)
   -> [Grad-CAM] -> heatmap overlay
   -> [Temperature scaling] -> calibrated probabilities
   -> [OOD detector] -> confidence flag
   -> Structured findings JSON {label, probability, calibrated_confidence, uncertain, region}
   -> LLM formatting layer (Claude API) -> prose report + disclaimer
   -> Gradio demo -> image + heatmap overlay + report
```

Two independent classifier heads (CXR, dermoscopy) share one downstream
pipeline (Grad-CAM -> calibration -> OOD -> LLM report). See
`models/backbone.py`, `explain/gradcam.py`, `calibrate/temperature_scaling.py`,
`ood/detectors.py`, `report/`.

## Repo layout

| Path | Contents |
|---|---|
| `data/` | Dataset classes, download helpers, label-prep scripts |
| `models/` | timm backbone factory, modality auto-router |
| `train/` | Training loop, checkpoint-resume, engine |
| `explain/` | Grad-CAM wrapper |
| `calibrate/` | Temperature scaling, ECE, reliability diagrams |
| `ood/` | Max-softmax/entropy + Mahalanobis OOD detectors |
| `report/` | Findings JSON schema, Claude API report generation |
| `app/` | Inference pipeline glue + Gradio demo |
| `eval/` | AUC table, ECE plots, OOD AUROC, failure gallery |
| `notebooks/` | Colab (CXR) / Kaggle (derm) training notebook templates |
| `configs/` | YAML config + loader |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # optional -- app falls back to a templated report without it
```

### 1. Get data (start this on day 1 -- access lead time is the main schedule risk)

```bash
# NIH ChestX-ray14 -- instant, no request needed
python -m data.download nih --dest data_raw/nih_cxr
python -m data.prepare_nih --root data_raw/nih_cxr

# ISIC dermoscopy -- instant, no login
python -m data.download isic --dest data_raw/isic --limit 8000
python -m data.prepare_isic --root data_raw/isic

# CheXpert -- submit the Stanford AIMI request form today, it's the only gated one
python -m data.download chexpert-instructions
```

CheXpert's CSV already matches the format `MultiLabelImageDataset` expects
(including the u/0/1/-1 uncertainty labels) -- once approved, just point
`configs/default.yaml`'s `data.cxr.labels_csv` / `data.cxr.root` at it.

### 2. Train

```bash
python -m train.train --modality cxr --set train.epochs=24   # or run notebooks/colab_train_cxr.ipynb on Colab
python -m train.train --modality derm                        # or run notebooks/kaggle_train_derm.ipynb on Kaggle
```

Training uses a cosine LR schedule with linear warmup by default
(`train.scheduler`/`train.warmup_epochs`/`train.min_lr` in
`configs/default.yaml`) -- it's a pure function of `global_step`, so it needs
no extra checkpoint state and resumes correctly across restarts. CXR
specifically benefits from more than the 8 epochs used in the first pass
(loss was still decreasing) -- see results below.

Training checkpoints every `train.checkpoint_every_steps` steps and
auto-resumes from `train.checkpoint_dir/<modality>_latest.pt` if present --
safe to kill and restart at any point (Colab disconnects, etc).

### 3. Calibrate + evaluate

```bash
python -m calibrate.fit_temperature --modality cxr
python -m eval.run_eval --modality cxr --ood-dataset path/to/ood_images/
```

`--ood-dataset` per the plan: feed it CIFAR/random web images, or the
*other* modality's images (derm images through the CXR model and vice
versa), to get OOD AUROC/FPR95 numbers.

### 4. Run the demo

```bash
python -m app.gradio_app
```

## Evaluation plan

- Per-pathology AUC (`eval/run_eval.py` -> `auc_table.md`), benchmarked
  against published CheXpert/NIH14 baselines.
- Calibration: ECE + reliability diagrams before/after temperature scaling.
- OOD detection: AUROC / FPR@95TPR.
- Grad-CAM sanity checks: qualitative, do heatmaps localize to clinically
  plausible regions (e.g. cardiomegaly -> heart).
- Failure case gallery: highest-confidence wrong predictions, curated with
  commentary (`eval/run_eval.py` -> `failure_gallery.md`).

### Results so far: dermoscopy branch (real data, real training run)

Trained end-to-end on this repo, on a MacBook (Apple M4 Pro, MPS backend --
no CUDA needed) against a real, class-balanced ISIC Archive sample (1,200
melanoma / 1,200 nevus / 1,200 other, pulled via `data.download isic`, since
plain unfiltered ISIC pulls turn out to be >97% nevus). DenseNet121, 8
epochs, ~35 min total.

| Label | AUC |
|---|---|
| melanoma | 0.870 |
| nevus | 0.957 |
| other | 0.907 |

- Test accuracy: 75.7% (macro AUC 0.911) on the 3-way held-out test split.
- Calibration: ECE 0.080 -> 0.038 after temperature scaling (T=0.84).
- OOD detection, real ISIC vs. real CIFAR-10 images -- entropy vs.
  Mahalanobis (`ood/detectors.py::MahalanobisDetector`, fit on penultimate
  features from the training set):

  | Method | AUROC | FPR@95TPR |
  |---|---|---|
  | entropy | 0.859 | 0.734 |
  | Mahalanobis | **0.9999** | **0.0** |

  Mahalanobis is now the default (`ood.method` in `configs/default.yaml`).
- Failure gallery: the highest-confidence misses are almost all
  melanoma<->nevus confusions -- the model is making the same mistake a
  human would find hardest, not a nonsensical one. See
  `eval_out/derm/failure_gallery.md`.

### Results so far: CXR branch (real data, real training run)

Trained against the official NIH ChestX-ray14 *sample* release (5,606
images -- the practical choice over the full ~42GB/112K-image dataset for a
laptop-scale run; `data.download nih` grabs the full one if you have the
time/bandwidth). DenseNet121, on the 5 "CheXpert competition" pathologies.

First pass was 8 epochs at a flat LR (~15 min), which the README originally
reported at test mean AUC 0.703 with loss still visibly decreasing. Second
pass added a cosine LR schedule with linear warmup (`train.scheduler:
cosine` in `configs/default.yaml`) and resumed that same checkpoint out to
24 epochs (~35 min more, ~50 min total):

| Label | AUC (8 epochs, flat LR) | AUC (24 epochs, cosine LR) |
|---|---|---|
| Atelectasis | 0.681 | 0.696 |
| Cardiomegaly | 0.604 | 0.632 |
| Consolidation | 0.762 | 0.777 |
| Edema | 0.803 | 0.809 |
| Pleural Effusion | 0.665 | 0.692 |

- Test mean AUC: 0.703 -> **0.722** after the longer cosine-scheduled run --
  still below the plan's ~0.80+ target bar (that gap traces to training-set
  size, 5,606 images vs. the full 112K, not the schedule), but a real,
  measured improvement from the same architecture and data, just more
  epochs and a decaying LR instead of a flat one.
- Calibration: temperature scaling still only helps modestly on this
  undertrained-relative-to-target model (e.g. Cardiomegaly ECE 0.400 ->
  0.321; Pleural Effusion barely moves, 0.354 -> 0.354) -- consistent with
  the first pass's finding that temperature scaling rescales confidence
  uniformly, it can't fix a model that needs more signal, and that still
  shows up honestly here.
- OOD detection, real CXR vs. real CIFAR-10 images -- entropy thresholding
  is still measurably broken on this model (AUROC 0.061, essentially
  unchanged from the first pass's 0.052), for the same reason as before: an
  undertrained classifier produces saturated, overconfident logits on
  out-of-distribution images, so entropy scores those *more* confident than
  real chest X-rays. Mahalanobis distance on penultimate features remains
  unaffected by this and stays at a clean AUROC 1.0 / FPR@95TPR 0.0.
- Failure gallery: `eval_out/cxr/failure_gallery.md`.

**A bug the retrain surfaced:** the cosine schedule initially returned a
`numpy.float64` (from `np.cos`) instead of a plain Python `float`, which
leaked into the optimizer's `param_group["lr"]` and then into the saved
checkpoint's `optimizer_state`. That tripped PyTorch 2.6+'s stricter
`weights_only=True` default on `torch.load`, breaking `calibrate.fit_temperature`
and `eval.run_eval` (but not training/resume itself, which had already
loaded the *old* checkpoint before this one existed). Fixed by casting to
`float()` in `train.engine.cosine_lr` and by making the checkpoint-loading
call sites (`train/checkpoint.py`, `calibrate/fit_temperature.py`,
`eval/run_eval.py`, `app/load.py`) explicit about `weights_only=False` --
correct here since these are checkpoints this repo trained and saved
itself, not third-party files.

Both branches' numbers come from real training runs in this repo, not
placeholders -- see [Status / what's left](#status--whats-left) for what a
longer run (full dataset, more epochs) would still need.

## Responsible AI / guardrails

- Persistent "research demo, not a diagnostic device" disclaimer in the UI
  (`app/gradio_app.py`) and in every generated report
  (`report/claude_report.py::DISCLAIMER_TEMPLATE`), enforced twice: once via
  the system prompt, once via a hard string check that re-prepends the
  disclaimer if the model ever drifts from it.
- Confidence/OOD flags surface as explicit "uncertain, do not rely on this"
  language (`report/build_findings.py`), not buried in a probability number.
- No PHI: only public, de-identified datasets (NIH14, CheXpert, ISIC).
- License compliance: CheXpert is non-commercial research use only; ISIC
  terms and NIH14 usage terms apply -- read them before any use beyond this
  portfolio project.
- Audit trail: every inference is logged (input hash, prediction,
  confidence, flags) to `app.audit_log_path` (`app/pipeline.py::log_inference`) --
  a stand-in for what a real deployment would need at production scale, with
  proper access controls and retention policy this demo does not implement.

## Status / what's left

- **Both branches: trained, calibrated, evaluated** on real data, real
  checkpoints, not placeholders:
  - Dermoscopy: 3,600-image ISIC sample. `checkpoints/derm_best.pt`,
    `derm_temperature.pt`, `derm_ood_threshold.json`, `derm_mahalanobis.npz`.
  - CXR: 5,606-image NIH ChestX-ray14 *sample* release.
    `checkpoints/cxr_best.pt`, `cxr_temperature.pt`, `cxr_ood_threshold.json`,
    `cxr_mahalanobis.npz`.
  - Both verified end-to-end through the full app pipeline (modality
    routing, Grad-CAM, calibration, Mahalanobis OOD flagging, report
    generation) and through an actual running Gradio server, not just unit
    tests.
- **OOD detection upgraded to Mahalanobis** (`ood/detectors.py::MahalanobisDetector`),
  now the default (`ood.method` in `configs/default.yaml`) after entropy
  thresholding measured *worse than random* (AUROC 0.052) on the CXR branch
  -- see the results tables above for the before/after on both branches.
  `eval.run_eval` fits and saves it automatically; `app/load.py` falls back
  to entropy if the fitted `.npz` is missing.
- **CXR: cosine LR schedule + more epochs (8 -> 24) done** -- test mean AUC
  0.703 -> 0.722. See results above for the full before/after and the
  `weights_only` checkpoint-loading bug this surfaced and fixed along the
  way.
- **What a longer run would still improve, in priority order:**
  1. CXR on the full NIH ChestX-ray14 (`nih-chest-xrays/data`, ~42GB,
     112K images) or CheXpert (needs *your* Stanford AIMI request under
     your own identity -- `python -m data.download chexpert-instructions`
     prints the steps) instead of the 5,606-image sample -- the CXR AUCs
     (0.63-0.81) trace back to training-set size, not a design flaw in the
     pipeline.
  2. `MahalanobisDetector.extract_features` uses each CXR image's argmax
     label as a pseudo-class (multi-label has no single ground-truth class
     to group by) -- works in practice (AUROC 1.0 here) but a per-label
     one-vs-rest Gaussian would be the principled version if this needs to
     generalize beyond a demo.

## Limitations

- Classifier heads are trained per-modality on public research datasets with
  known label noise (NIH14 labels are NLP-mined from radiology reports, not
  expert-adjudicated) -- reported AUCs (see results above, or
  `eval_out/<modality>/auc_table.md`) are not clinical-grade performance,
  and the CXR numbers specifically reflect a 5,606-image sample rather than
  the full 112K-image dataset.
- Modality routing (`models/modality_router.py`) is a saturation heuristic,
  not a learned classifier -- it will misroute unusual inputs (e.g. a
  grayscale dermoscopy image, a color-annotated CXR).
- OOD detection catches inputs statistically unlike the training
  distribution; it does not catch in-distribution-looking inputs from a
  different clinical context (different scanner, population, disease
  prevalence) -- calibration and OOD numbers do not transfer across sites
  without revalidation.
- Grad-CAM highlights *where* the model attended, not *why* -- it is a
  sanity-check tool for developers, not an explanation a patient or
  clinician should interpret as ground truth.

## License

Code in this repo: for the license text, see `LICENSE`. Datasets keep their
own terms regardless of this repo's license -- CheXpert (Stanford, research/
non-commercial), ISIC (CC-0 / per-image, check the archive), NIH ChestX-ray14
(NIH usage terms) -- do not redistribute the raw data.
