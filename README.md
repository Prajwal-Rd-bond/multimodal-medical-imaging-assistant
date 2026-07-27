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

This branch has gone through three training passes, each real, each on this
same 5,606-image sample -- the improvements below come from fixing the
pipeline, not from more data:

1. **8 epochs, flat LR:** test mean AUC 0.703, loss still visibly decreasing.
2. **24 epochs, cosine LR + warmup** (`train.scheduler: cosine` in
   `configs/default.yaml`), resumed from the same checkpoint: test mean AUC
   **0.722**.
3. **Same 24-epoch cosine run, but from scratch, after fixing a backbone-freeze
   bug** (below) that had made passes 1-2 linear probes on frozen ImageNet
   features rather than actual fine-tuning, plus a focal loss for the class
   imbalance and slightly stronger augmentation: test mean AUC **0.782**
   (best checkpoint, epoch 15/24).

| Label | AUC (linear probe, 24ep) | AUC (real fine-tune, 24ep) |
|---|---|---|
| Atelectasis | 0.696 | 0.728 |
| Cardiomegaly | 0.632 | **0.840** |
| Consolidation | 0.777 | 0.709 |
| Edema | 0.809 | 0.798 |
| Pleural Effusion | 0.692 | 0.833 |

**The bug:** `models/backbone.py::ClassifierHead._freeze_stages` froze the
first `freeze_backbone_stages` *top-level children* of the backbone. For
DenseNet121 in `timm` with `num_classes=0`, the backbone has exactly one
top-level child that matters -- `features`, a single `Sequential` bundling
every conv stage (stem through `denseblock4`/`norm5`). So the config's
`freeze_backbone_stages: 2` (originally meant as "freeze the first couple
of generic-feature stages") froze `features` in its entirety on iteration
0 -- i.e. **all 6.95M conv parameters**, in every CXR run before this one.
Only the small `Linear` head was ever training. Fixed by walking one level
into `backbone.features` so `freeze_stages` selects real DenseNet stages
(conv0/norm0/pool0/denseblock1/.../norm5, 11 total); default is now 4
(freeze the stem + denseblock1, fine-tune denseblock2 onward). Confirmed
with a quick trainable-parameter count before shipping the fix:
`freeze_stages=2` under the old code left 6,949,445/6,958,981 params
trainable-*looking* but actually froze the feature extractor; under the
new code it correctly leaves only ~9.5K (the stem) frozen. Cardiomegaly --
the pathology that most needs texture/shape features an ImageNet backbone
was never going to have -- shows the effect most clearly: 0.632 -> 0.840.

**Also added this pass:**
- Focal loss (`train.focal_gamma` in `configs/default.yaml`, default 2.0)
  on top of the existing per-class `pos_weight`: down-weights already-easy
  examples so the loss concentrates on hard, usually-minority positives.
  `train/engine.py::_multilabel_loss`.
- Slightly stronger augmentation (`data/transforms.py::train_transforms`):
  `RandomResizedCrop(scale=(0.85, 1.0))` instead of a plain resize, plus a
  bit more rotation/jitter -- modest translation/scale augmentation for a
  training set this small.
- A `weights_only` checkpoint-loading bug the cosine-schedule pass (#2
  above) introduced and this pass fixed: `cosine_lr` returned a
  `numpy.float64` (from `np.cos`) instead of a plain Python `float`, which
  leaked into the optimizer's `param_group["lr"]` and then into the saved
  checkpoint's `optimizer_state`, tripping PyTorch 2.6+'s stricter
  `weights_only=True` default on load. Fixed by casting to `float()` in
  `train.engine.cosine_lr`, and by making the checkpoint-loading call sites
  (`train/checkpoint.py`, `calibrate/fit_temperature.py`,
  `eval/run_eval.py`, `app/load.py`) explicit about `weights_only=False` --
  correct here since these are checkpoints this repo trained and saved
  itself, not third-party files.

Calibration and OOD after the fine-tune fix:

- ECEs roughly halved pre-calibration vs. the linear-probe run, consistent
  with a model that's actually learned the task rather than reweighting
  frozen features (e.g. Cardiomegaly 0.400 -> 0.151, Edema 0.351 -> 0.156),
  and temperature scaling still helps further on top of that (Cardiomegaly
  0.151 -> 0.094, Edema 0.156 -> 0.105).
- OOD detection, real CXR vs. real CIFAR-10 images: entropy thresholding is
  *still* broken (AUROC 0.096) -- an accurate-on-CXR model can still
  produce saturated logits on natural images it's never seen, so this isn't
  purely an "undertrained model" artifact, it's a property of entropy
  thresholding on out-of-distribution natural images generally. Mahalanobis
  distance on penultimate features is unaffected either way: AUROC 1.0 /
  FPR@95TPR 0.0.
- Failure gallery: `eval_out/cxr/failure_gallery.md`.

Both branches' numbers come from real training runs in this repo, not
placeholders -- see [Status / what's left](#status--whats-left) for what a
longer run (full dataset, more epochs) would still need. The old
linear-probe-era checkpoints are kept at
`checkpoints/frozen_backbone_baseline/` for comparison, not used by the app.

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
- **CXR: cosine LR schedule, then a real backbone-freeze bug fix, done** --
  test mean AUC 0.703 -> 0.722 (schedule + more epochs) -> **0.782** (fixing
  `models/backbone.py::_freeze_stages`, which had left every earlier CXR
  run training a linear probe on frozen ImageNet features, not actually
  fine-tuning -- see results above for the full history, before/afters, and
  the two bugs this surfaced and fixed along the way).
- **Dermoscopy has the same latent freeze bug, unfixed:** `derm`'s
  `freeze_backbone_stages: 2` (now correctly 4 as of this fix, applied
  repo-wide) means the 0.911 macro AUC reported above is *also* a
  frozen-backbone linear-probe result, not a real fine-tune -- it just
  happened to work reasonably well because ImageNet features transfer to
  skin-lesion color/texture better than to chest X-ray structure. Retraining
  derm the same way (real fine-tune + focal loss where relevant) would
  likely improve it further and hasn't been done yet.
- **What would still improve CXR further, in priority order:**
  1. CXR on the full NIH ChestX-ray14 (`nih-chest-xrays/data`, ~42GB,
     112K images) or CheXpert (needs *your* Stanford AIMI request under
     your own identity -- `python -m data.download chexpert-instructions`
     prints the steps) instead of the 5,606-image sample -- more data on
     top of a now-correctly-fine-tuned model, rather than instead of one.
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
