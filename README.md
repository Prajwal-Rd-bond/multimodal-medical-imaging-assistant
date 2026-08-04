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
| `server/` | FastAPI backend for the standalone web frontend (same pipeline, JSON API) |
| `web/` | React + Vite + TypeScript frontend (polished UI, talks to `server/`) |
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

# NIH ChestX-ray14, more than the 5,606-image sample without the full 42GB:
# `data.download nih` only offers "sample" (5,606 img) or "full" (42GB) via
# Kaggle, which bundles the full dataset as one archive. The
# alkzar90/NIH-Chest-X-ray-dataset Hugging Face mirror splits it into the
# same 12 per-batch zips NIH itself uses, so you can pull just as many
# batches as your disk/bandwidth budget allows (each ~2-4GB, ~9,300 images):
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, zipfile
repo = 'alkzar90/NIH-Chest-X-ray-dataset'
shutil.copy(hf_hub_download(repo, repo_type='dataset', filename='data/Data_Entry_2017_v2020.csv'), 'data_raw/nih_cxr/Data_Entry_2017.csv')
for i in range(1, 4):  # batches 1-3 =~ 9.89GB; add more up to 12 for the full dataset
    p = hf_hub_download(repo, repo_type='dataset', filename=f'data/images/images_{i:03d}.zip')
    zipfile.ZipFile(p).extractall('data_raw/nih_cxr')
"
mkdir -p data_raw/nih_cxr/images_hf && mv data_raw/nih_cxr/images data_raw/nih_cxr/images_hf/images  # match the images_*/images layout data.prepare_nih expects
python -m data.prepare_nih --root data_raw/nih_cxr  # rebuilds labels.csv against everything now on disk

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

### 5. Run the full-stack web app (custom UI, instead of Gradio)

Same inference pipeline (`app/pipeline.py`, `app/load.py`), a different front
door: a FastAPI JSON API (`server/`) plus a standalone React + Vite +
TypeScript + Tailwind frontend (`web/`) with a custom, animated UI (drag-drop
upload, Grad-CAM/original toggle, calibrated-confidence bars, OOD/uncertainty
flags, light + dark theme). Two terminals:

```bash
# Terminal 1 -- backend, http://localhost:8000
python -m server.main

# Terminal 2 -- frontend, http://localhost:5173
cd web && npm install && npm run dev
```

Open `http://localhost:5173`. The frontend calls the backend at
`http://localhost:8000` by default; override with `VITE_API_BASE` (frontend)
and `CORS_ORIGINS` (backend, comma-separated) if you deploy them on different
hosts. The Gradio app (`app/gradio_app.py`, step 4 above) still works
unchanged and is what Hugging Face Spaces launches per this README's
frontmatter -- the two front doors are independent, pick either.

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
plain unfiltered ISIC pulls turn out to be >97% nevus). DenseNet121.

Like the CXR branch (see above), the first derm run also trained under the
`_freeze_stages` bug -- `freeze_backbone_stages: 2` froze DenseNet121's
entire `features` block, so this was a linear probe on frozen ImageNet
features, not a fine-tune. It happened to score reasonably (0.911 macro
AUC) because ImageNet features transfer to skin-lesion color/texture better
than to chest-X-ray structure, which masked the bug here for longer than on
CXR. After fixing `models/backbone.py::_freeze_stages` (real fine-tuning,
denseblock2 onward) and retraining from scratch, 24 epochs, cosine LR:

| Label | AUC (linear probe, 8ep) | AUC (real fine-tune, 24ep) |
|---|---|---|
| melanoma | 0.870 | 0.953 |
| nevus | 0.957 | 0.981 |
| other | 0.907 | 0.979 |

- Test accuracy: 75.7% -> **87.4%** (macro AUC 0.911 -> **0.969**) on the
  3-way held-out test split.
- Calibration: ECE (pre-scaling) 0.080 -> 0.050, and temperature scaling
  still helps on top of that: 0.050 -> **0.015** (T=1.49). (Updated after
  fixing `calibrate.max_iter` -- see the CXR calibration writeup below for
  the full story; this number was originally reported as 0.032, from a
  scaler that hadn't actually converged.)
- OOD detection, real ISIC vs. real CIFAR-10 images -- entropy vs.
  Mahalanobis (`ood/detectors.py::MahalanobisDetector`, fit on penultimate
  features from the training set):

  | Method | AUROC | FPR@95TPR |
  |---|---|---|
  | entropy | 0.858 | 0.656 |
  | Mahalanobis | **0.99997** | **0.0** |

  Mahalanobis is now the default (`ood.method` in `configs/default.yaml`).
  Both methods land close to the linear-probe run's numbers -- unlike CXR,
  entropy thresholding was never badly broken here, it's just consistently
  worse than Mahalanobis.
- Failure gallery: the highest-confidence misses are still almost all
  melanoma<->nevus confusions -- the model is making the same mistake a
  human would find hardest, not a nonsensical one. See
  `eval_out/derm/failure_gallery.md`. Old linear-probe checkpoints are kept
  at `checkpoints/frozen_backbone_baseline/` for comparison.

### Results so far: CXR branch (real data, real training run)

Trained against the official NIH ChestX-ray14 dataset, DenseNet121, on the 5
"CheXpert competition" pathologies.

This branch has gone through four training passes, each real:

1. **8 epochs, flat LR, 5,606-image sample:** test mean AUC 0.703, loss
   still visibly decreasing.
2. **24 epochs, cosine LR + warmup** (`train.scheduler: cosine` in
   `configs/default.yaml`), same 5,606-image sample, resumed from the same
   checkpoint: test mean AUC **0.722**.
3. **Same 24-epoch cosine run, but from scratch, after fixing a backbone-freeze
   bug** (below) that had made passes 1-2 linear probes on frozen ImageNet
   features rather than actual fine-tuning, plus a focal loss for the class
   imbalance and slightly stronger augmentation, still on the 5,606-image
   sample: test mean AUC **0.782** (best checkpoint, epoch 15/24).
4. **Same fine-tuning setup, retrained on 29,355 images** -- the 5,606-image
   sample plus NIH ChestX-ray14 batches `images_001`-`images_003` (24,999
   more images, ~9.89GB, pulled from the `alkzar90/NIH-Chest-X-ray-dataset`
   Hugging Face mirror rather than the Kaggle copy, since Kaggle only offers
   the full ~42GB dataset as one bundle and the HF mirror splits it into the
   same 12 per-batch zips the original NIH release uses -- a deliberate
   partial-data choice to stay within an ~11GB budget rather than pull the
   full 112K-image set): test mean AUC **0.826** (best checkpoint, epoch
   9/24; test set matches the images actually downloaded, so the earlier
   5,606-image sample's own held-out split changed too -- not a strictly
   apples-to-apples test set, but AUC is stable in the low-0.8 range across
   both).

| Label | AUC (linear probe, 24ep) | AUC (fine-tune, 5,606 img) | AUC (fine-tune, 29,355 img) |
|---|---|---|---|
| Atelectasis | 0.696 | 0.728 | 0.777 |
| Cardiomegaly | 0.632 | 0.840 | **0.921** |
| Consolidation | 0.777 | 0.709 | 0.712 |
| Edema | 0.809 | 0.798 | **0.873** |
| Pleural Effusion | 0.692 | 0.833 | **0.848** |

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

Calibration and OOD after the fine-tune fix (5,606-image run):

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

Calibration and OOD after the 29,355-image retrain:

- First measurement (`calibrate.max_iter: 50`, the config default at the
  time) showed ECE *worse* than the 5,606-image run despite better AUC --
  e.g. Cardiomegaly post-scaling ECE 0.094 (5,606-image run) vs. 0.176
  (29,355-image run). That looked like a real regression (more/harder data
  hurting calibration) but wasn't: `calibrate/fit_temperature.py` fits the
  per-class temperature with LBFGS in a single `optimizer.step(closure)`
  call, and 50 iterations was enough to converge on the small validation
  set but not the larger one. Comparing `max_iter` 50/200/500/2000 (holding
  everything else fixed) showed ECE still improving through 500 and
  identical at 500 vs. 2000 -- i.e. 50 was silently under-converged, not a
  data-scale effect. Fixed by bumping the default to 500 (`configs/default.yaml`).
  The same bug was quietly hurting the **dermoscopy** branch too (see its
  results above) -- it just looked fine there because temperature scaling's
  under-converged result (ECE 0.032) still beat the pre-scaling baseline
  (0.050), so nothing looked broken, the same way the derm branch's
  frozen-backbone bug hid longer than CXR's.
- ECE (pre-scaling -> post-scaling, with the fix): Atelectasis 0.320 ->
  0.274, Cardiomegaly 0.215 -> **0.142**, Consolidation 0.309 -> 0.237, Edema
  0.163 -> **0.122**, Pleural Effusion 0.257 -> 0.170 (T=[0.52, 0.51, 0.49,
  0.64, 0.43]). Still somewhat higher than the 5,606-image run's post-scaling
  numbers (Cardiomegaly 0.094, Edema 0.105) -- unlike the convergence gap
  above, that remaining difference hasn't been root-caused, and calling it
  "more data made calibration harder" would be exactly the unverified claim
  that caused this bug in the first place, so it's left as an open question
  rather than a confident explanation.
- OOD detection, real CXR vs. real CIFAR-10 images: entropy thresholding is
  still broken, slightly worse than before (AUROC 0.096 -> 0.067) -- same
  story as the 5,606-image run, an accurate-on-CXR model still produces
  saturated logits on natural images it's never seen. Mahalanobis distance
  on penultimate features is unaffected: AUROC 1.0 / FPR@95TPR 0.0.
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
- Grad-CAM overlay is only presented as a real finding when a label actually
  clears the classification threshold (`app/pipeline.py`'s `overlay_confident`
  flag, threaded through both the Gradio app and the web UI). Grad-CAM always
  *produces* a heatmap even for a label the model never called positive
  (`explain_top_k`'s "always explain at least the top prediction" fallback,
  needed so there's something to show), but gradients for a low-confidence
  logit are weak and noisy and routinely light up anatomically meaningless
  regions -- bony edges, image borders -- rather than anything real (verified
  by comparing heatmaps for confident vs. unconfident labels on the same
  image: a 98%-confidence Cardiomegaly call centers cleanly on the heart, a
  <30%-confidence call on the same image lights up the shoulder). Presenting
  that fallback heatmap with the same visual weight as a confident one is
  actively misleading, so the UI now shows an explicit low-confidence warning
  instead of "highlighting X" language when nothing cleared the threshold.
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
  - CXR: 29,355-image NIH ChestX-ray14 (the original 5,606-image sample
    plus batches `images_001`-`images_003` from the
    `alkzar90/NIH-Chest-X-ray-dataset` Hugging Face mirror, ~11GB).
    `checkpoints/cxr_best.pt`, `cxr_temperature.pt`, `cxr_ood_threshold.json`,
    `cxr_mahalanobis.npz`. The old 5,606-image-only checkpoints are kept at
    `checkpoints/cxr_5k_sample_baseline/` for comparison.
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
- **CXR: cosine LR schedule, then a real backbone-freeze bug fix, then 5x more
  data, done** -- test mean AUC 0.703 -> 0.722 (schedule + more epochs) ->
  0.782 (fixing `models/backbone.py::_freeze_stages`, which had left every
  earlier CXR run training a linear probe on frozen ImageNet features, not
  actually fine-tuning) -> **0.826** (retraining the same fixed setup on
  29,355 images instead of 5,606) -- see results above for the full history,
  before/afters, and the two bugs this surfaced and fixed along the way.
  Calibration initially looked *worse* on the bigger run (ECE up despite AUC
  up) -- turned out to be a genuine bug (`calibrate.max_iter: 50` wasn't
  enough for LBFGS to converge on the larger validation set, silently
  affecting the dermoscopy branch too), now fixed; see the calibration
  writeup above for the full story and the numbers after the fix.
- **Grad-CAM overlay no longer presented as a finding when unconfident** --
  the app used to always show a heatmap for the top predicted label, even
  when nothing crossed the classification threshold; those heatmaps are
  gradient noise (they light up bony edges, not pathology) and looked like a
  real localized finding. Both front ends (Gradio, web UI) now show an
  explicit low-confidence warning instead. See the guardrails section above.
- **Dermoscopy retrained with the same freeze-stages fix:** macro AUC 0.911
  -> **0.969**, accuracy 75.7% -> 87.4%. The original 0.911 was also a
  frozen-backbone linear-probe result -- it just scored reasonably because
  ImageNet features transfer to skin-lesion color/texture better than to
  chest X-ray structure, which masked the bug here longer than on CXR. See
  results above for the full before/after.
- **What would still improve CXR further, in priority order:**
  1. More CXR data still helps: the 5,606 -> 29,355-image retrain is proof
     of that (0.782 -> 0.826 test mean AUC on the exact same architecture/
     hyperparameters), so the rest of NIH ChestX-ray14 (batches
     `images_004`-`images_012`, another ~30GB/~87K images via the same
     `alkzar90/NIH-Chest-X-ray-dataset` HF mirror) or CheXpert (needs *your*
     Stanford AIMI request under your own identity -- `python -m
     data.download chexpert-instructions` prints the steps) is the next
     lever to pull, now that the model is correctly fine-tuning.
  2. Investigate the calibration regression on the 29,355-image run (ECE
     worse than the 5,606-image run despite better AUC -- see the
     calibration writeup above) before trusting `calibrated_confidence` at
     face value on this checkpoint. More epochs, or an LR schedule tuned for
     the larger dataset, are the first things to try.
  3. `MahalanobisDetector.extract_features` uses each CXR image's argmax
     label as a pseudo-class (multi-label has no single ground-truth class
     to group by) -- works in practice (AUROC 1.0 here) but a per-label
     one-vs-rest Gaussian would be the principled version if this needs to
     generalize beyond a demo.

## Limitations

- Classifier heads are trained per-modality on public research datasets with
  known label noise (NIH14 labels are NLP-mined from radiology reports, not
  expert-adjudicated) -- reported AUCs (see results above, or
  `eval_out/<modality>/auc_table.md`) are not clinical-grade performance,
  and the CXR numbers specifically reflect a 29,355-image subset (the
  original 5,606-image sample plus 3 of NIH ChestX-ray14's 12 batches)
  rather than the full 112K-image dataset.
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
