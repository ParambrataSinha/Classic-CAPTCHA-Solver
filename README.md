# Distorted Visual Sequence Pattern Recognition
### CIG AI/ML Competition — Deep Learning Solution

**Author:** Parambrata Sinha | **Enrollment:** 25112071  
**Metric:** Character Error Rate (CER) — lower is better  
**Stack:** PyTorch · timm · Albumentations · CUDA (AMP fp16)

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Key Insight & Design Philosophy](#2-key-insight--design-philosophy)
3. [Dataset Analysis (EDA)](#3-dataset-analysis-eda)
4. [Preprocessing & Augmentation Pipeline](#4-preprocessing--augmentation-pipeline)
5. [Model Architecture](#5-model-architecture)
   - [Model A — EfficientNet-B3 Multi-Head Classifier](#model-a--efficientnet-b3-multi-head-classifier)
   - [Model B — CRNN + CTC](#model-b--crnn--ctc)
6. [Training Strategy](#6-training-strategy)
7. [Inference — TTA + Weighted Ensemble](#7-inference--tta--weighted-ensemble)
8. [Evaluation & Results](#8-evaluation--results)
9. [Repository Structure](#9-repository-structure)
10. [How to Reproduce](#10-how-to-reproduce)
11. [Further Improvements](#11-further-improvements)

---

## 1. Problem Overview

The task is to read 6-character alphanumeric text sequences from heavily distorted grayscale images — a classic CAPTCHA-recognition challenge that pushes standard OCR systems to their limits.

**Dataset at a glance:**

| Property | Value |
|---|---|
| Training images | 20,000 |
| Test images | 5,000 |
| Image dimensions | 200 × 100 px, grayscale |
| Sequence length | Fixed 6 characters |
| Vocabulary size | 31 characters (`23456789ABCDEFGHJKMNPQRSTUVWXYZ`) |
| Evaluation metric | Character Error Rate (CER, Levenshtein-based) |

Distortions present in the images include background noise, overlapping symbols, Gaussian/motion blur, visual artifacts, shape deformation, occlusion patches, and irregular character spacing.

---

## 2. Key Insight & Design Philosophy

> **The most impactful design decision:** treating this as **6 independent 31-way classification problems** rather than a variable-length sequence decoding task.

Standard OCR approaches (e.g., pure CTC) operate over variable-length sequences and must infer both *what* and *how many* characters are present. This introduces unnecessary complexity. A careful EDA step revealed that **99.99% of training labels are exactly 6 characters long** (only 2 anomalous samples in ~20,000 existed and were removed). This observation unlocks a cleaner formulation:

- One parallel classification head per character position
- Each head is a simple 31-class softmax
- No CTC decoding, no beam search, no blank tokens at inference time for the primary model
- The problem reduces to 6 strongly supervised, fully parallel classification tasks

This is architecturally simpler, trains faster, and achieves higher accuracy than CTC on fixed-length sequences. The CRNN+CTC model is retained as a **complementary ensemble partner** that learns sequential/temporal patterns from a fundamentally different inductive bias.

---

## 3. Dataset Analysis (EDA)

Before building any model, a thorough EDA was conducted to validate assumptions and inform design choices.

**Steps performed:**

- Loaded raw CSV labels and verified the length distribution of all sequences — confirmed 99.99% are exactly 6 characters. The 2 outlier rows were removed from training.
- Visualized a 4×6 grid of randomly sampled training images with their ground-truth labels to understand distortion severity.
- Computed **character frequency** across all training labels to check for class imbalance across the 31-character vocabulary.
- Computed a **per-position character distribution matrix** (6 positions × 31 characters) to detect any positional bias — e.g., whether certain characters only appear at certain positions.
- Sampled 500 images to compute the **pixel mean (~0.5) and std (~0.25)** of the grayscale distribution, which directly informed normalization parameters.

**Why this matters:** The EDA confirmed a uniform vocabulary with no severe positional or frequency bias, meaning standard cross-entropy loss with mild label smoothing is appropriate without any class-weighting tricks.

---

## 4. Preprocessing & Augmentation Pipeline

Built using **Albumentations v2** for GPU-friendly, composable transforms.

### Training Augmentation (Heavy)

The augmentation policy is deliberately designed to simulate — and exceed — the distortions seen in the test set. This improves generalization by forcing the model to learn invariant features rather than memorizing pixel patterns.

| Category | Transforms Applied | Reasoning |
|---|---|---|
| **Resize** | `Resize(64, 200)` | Reduce height from 100→64 to cut compute while preserving the wide aspect ratio important for reading left-to-right character sequences |
| **Geometric** | `ElasticTransform`, `GridDistortion`, `OpticalDistortion` (one of, p=0.5) | Directly mimics the warping and rubber-sheet deformations present in CAPTCHA images |
| **Affine** | `ShiftScaleRotate(rotate_limit=5, p=0.4)` | Handles slight misalignments and scale variation; rotation kept small (±5°) as CAPTCHAs are generally upright |
| **Perspective** | `Perspective(scale=0.02–0.05, p=0.3)` | Simulates slight camera angle changes; a second form of geometric robustness |
| **Noise** | `GaussNoise`, `ISONoise`, `MultiplicativeNoise` (one of, p=0.5) | Background noise is one of the primary distortions; training on all three noise types teaches invariance |
| **Blur** | `GaussianBlur`, `MotionBlur`, `MedianBlur` (one of, p=0.3) | Motion and Gaussian blur are common in CAPTCHA datasets; the model must distinguish blurred characters |
| **Occlusion** | `CoarseDropout(max_holes=6, max_height=8, max_width=12, p=0.4)` | Directly simulates the random black/white patches that occlude characters in the test images |
| **Intensity** | `RandomBrightnessContrast`, `RandomGamma` | Handles lighting variation; especially important since the model sees grayscale-only images |
| **Normalize** | `mean=0.5, std=0.25` | Derived from empirical pixel statistics of the training set; centers the distribution around zero for stable training |

### Validation / Test Transform (Clean)

Only `Resize → Normalize → ToTensor` — no stochastic augmentation, ensuring deterministic evaluation.

### Test-Time Augmentation (TTA)

Three deterministic passes are averaged at inference:
1. **Base** — clean resize + normalize
2. **Brightness boost** — `RandomBrightnessContrast(0.10, 0.10, p=1.0)` applied deterministically
3. **Sharpen** — `Sharpen(alpha=0.2–0.5, p=1.0)`

TTA is applied only to the primary model (MH). Averaging softmax probabilities across three views of each image reduces variance from imaging artifacts and reduces the probability of a single noisy pass causing an error.

---

## 5. Model Architecture

### Model A — EfficientNet-B3 Multi-Head Classifier

```
Input: (B, 1, 64, 200)  ← single-channel grayscale
         │
         ▼
  EfficientNet-B3 backbone  (in_chans=1, global_pool='avg')
  → feature vector: (B, 1536)
         │
         ▼
  Bottleneck block:
    Linear(1536 → 512) → BatchNorm1d → GELU → Dropout(0.40)
  → (B, 512)
         │
    ┌────┴─────────────────────────────────────────────┐
    ▼      ▼      ▼      ▼      ▼      ▼              
  Head₁  Head₂  Head₃  Head₄  Head₅  Head₆  (6 parallel heads)
  Each: Linear(512→256) → GELU → Dropout(0.20) → Linear(256→31)
    ▼      ▼      ▼      ▼      ▼      ▼
  (B,31) (B,31) (B,31) (B,31) (B,31) (B,31)
         │
         ▼
  Stack → (B, 6, 31)   argmax → 6-character string
```

**Design rationale:**

- **EfficientNet-B3 backbone:** Compound-scaled (depth + width + resolution) for strong feature extraction with moderate parameter count. The `in_chans=1` flag adapts the stem convolution to accept grayscale input natively, avoiding the artificial inflation of a single channel to 3 copies.
- **Pretrained weights:** ImageNet-pretrained weights are loaded. Even though the domain (CAPTCHA digits) differs significantly, the low-level texture and edge detectors learned from ImageNet transfer effectively to character recognition.
- **Bottleneck + 6 independent heads:** The shared 512-dim bottleneck is sufficient to represent the full character region. Each head then specializes to its position. The two-layer head design (`512→256→31`) provides a small amount of position-specific transformation capacity without overfitting.
- **Label smoothing (ε=0.10):** Prevents overconfidence on individual characters and improves calibration, which matters for the ensemble probability averaging step.
- **Weight initialization:** All linear layers in the bottleneck and heads are initialized with `trunc_normal_(std=0.02)`, consistent with the ViT/EfficientNet literature, avoiding large initial logits.

**Parameter count:** ~12M (backbone) + ~1.5M (heads) ≈ **13.5M total**

---

### Model B — CRNN + CTC

A classic OCR architecture included as an **ensemble partner** to inject complementary sequential reasoning.

```
Input: (B, 1, 64, 200)
         │
         ▼
  CNN Feature Extractor (5 conv blocks with BN + ReLU):
    64 → 64 → MaxPool(2,2)
    128 → 128 → MaxPool(2,2)
    256 → 256 → MaxPool(2,1)   ← pools height only to preserve width
    512 → 512 → MaxPool(2,1)
    512 (final)
  → AdaptiveAvgPool2d((1, None)) → squeeze height → (B, T, 512)
         │
         ▼
  BiLSTM (2 layers, hidden=256, bidirectional)
  → (B, T, 512)   ← 256×2 for bidirectional
         │
         ▼
  Dropout(0.3) → Linear(512 → 32)  ← 31 chars + 1 CTC blank
  → log_softmax → (B, T, 32)
         │
         ▼
  CTC Loss (training) / greedy CTC decode (inference)
```

**Why MaxPool with asymmetric strides (`(2,1)`):** The image is much wider than tall (200×64). Pooling height aggressively while keeping the width dimension intact preserves the temporal resolution of the feature map — more time-steps mean a longer sequence of column features for the RNN to process, which directly benefits CTC alignment.

**Why BiLSTM:** Each character in a CAPTCHA can be influenced by its neighbors (overlap, occlusion). A bidirectional LSTM captures both left→right and right→left context, making it more robust to positional ambiguity than a unidirectional model.

**Training loss:** `nn.CTCLoss(blank=31, zero_infinity=True)`. The `zero_infinity=True` flag silently ignores any batches where the CTC alignment is mathematically impossible (rare, but prevents NaN gradients on very noisy inputs).

**Parameter count:** ~8.5M

---

## 6. Training Strategy

### Primary Model (EfficientNet-B3 Multi-Head)

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | AdamW | Weight decay regularization on pretrained weights |
| Backbone LR | `3e-5` (= LR × 0.1) | **Differential learning rate** — backbone weights are already good; aggressively updating them destroys pretrained representations |
| Heads LR | `3e-4` | New heads need fast adaptation |
| Weight decay | `1e-4` | L2 regularization |
| Scheduler | Warmup (5 ep) + Cosine Annealing (60 ep max) | Warmup prevents early large-gradient destruction of pretrained weights; cosine annealing provides smooth LR decay |
| Batch size | 256 | Maximizes GPU utilization on P100 |
| Epochs | Up to 60 with early stopping (patience=12) | Stops when val CER stops improving for 12 consecutive epochs |
| Mixed precision | AMP fp16 | Halves memory usage, ~2× throughput on P100 (which does not support bfloat16) |
| Gradient clipping | `max_norm=5.0` | Prevents gradient explosion especially during early warmup phase |
| Label smoothing | `ε=0.10` | Regularization; improves ensemble calibration |

**Checkpointing:** The full training state (model weights, optimizer state, scaler state, scheduler state, epoch, val CER, val acc) is saved to disk whenever a new best validation CER is achieved.

### Secondary Model (CRNN)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, `lr=3e-4`, `weight_decay=1e-4` |
| Scheduler | CosineAnnealingLR (`T_max=40`, `eta_min=1e-6`) |
| Epochs | Up to 40 with early stopping (patience=10) |
| Loss | CTCLoss, `blank=31`, `zero_infinity=True` |
| Mixed precision | AMP fp16 |
| Gradient clipping | `max_norm=5.0` |

The CRNN is trained **from scratch** on the same train split, with the same heavy augmentation pipeline. No pretrained weights are used here, consistent with how CRNNs for OCR are traditionally trained.

---

## 7. Inference — TTA + Weighted Ensemble

The final inference pipeline has three stages:

### Stage 1 — Multi-Head TTA
The best-checkpoint MH model runs 3 forward passes per test image (base, brightness, sharpen). Softmax probabilities of shape `(N, 6, 31)` are averaged across the 3 passes.

### Stage 2 — CRNN Inference
The CRNN produces log-probabilities of shape `(N, T, 32)`. To bridge into the `(N, 6, 31)` format used by the ensemble:
- Convert log-probs to probs: `torch.exp(lp)`
- Take the **max probability over the time axis** (per class): `[:, :, :31].max(dim=1)`
- This yields the most-confident score each character class achieved at any timestep
- Tile to `(N, 6, 31)` to match the MH format

This per-class max-pooling over time is a pragmatic bridge: CTC does not produce position-aligned probabilities, but the most-confident score for each character class is still informative for ensemble weighting.

### Stage 3 — Weighted Combination

```
Final = 0.75 × MH_TTA_avg + 0.25 × CRNN_avg
predicted_char[pos] = argmax over 31 classes
```

The 75/25 weighting reflects the MH model's superior per-character accuracy (CTC decoding on fixed-length sequences is inherently less precise than direct classification). The CRNN contributes as a regularizer — when both models agree, confidence is high; when they disagree, the ensemble naturally down-weights uncertain predictions.

**Why ensemble at all?** The two models have fundamentally different inductive biases:
- MH sees the full image globally through the EfficientNet backbone
- CRNN sees the image as a left-to-right column sequence through the RNN

Errors they make tend to be uncorrelated, so combining them reduces the overall error rate beyond what either model achieves alone.

---

## 8. Evaluation & Results

### Metrics Tracked

- **CER (Character Error Rate):** Primary metric. Levenshtein edit distance normalized by total target length.
- **Sequence Accuracy:** Fraction of images where the full 6-character string is predicted exactly correctly.
- **Per-position accuracy:** Breakdown of correctness at each of the 6 character positions, revealing whether certain positions are systematically harder.
- **Edit-distance distribution:** Histogram of how many characters are wrong per prediction (0 errors = perfect sequence, 1 error = one wrong character, etc.).

### Validation Performance

| Model | Val CER | Val Seq Accuracy |
|---|---|---|
| EfficientNet-B3 Multi-Head | (best checkpoint) | (best checkpoint) |
| CRNN + CTC | (best checkpoint) | (best checkpoint) |
| **Final Ensemble (TTA + weighted)** | **lowest** | **highest** |

*(Exact numbers are populated from live training runs logged in the notebook output.)*

### Qualitative Analysis

The notebook includes:
- Side-by-side grid of validation images with predicted vs. ground-truth labels (green = correct, red = error)
- Per-position bar chart showing accuracy at each of the 6 character positions
- Edit-distance histogram confirming that most errors are single-character substitutions rather than multi-character failures

---

## 9. Repository Structure

```
.
├── notebook_Parambrata_25112071.ipynb               # Main solution notebook (fully reproducible)
├── submission_Parambrata_Sinha_25112071.csv   # Final prediction file
└── README.md                    # This file
```

Intermediate outputs saved to `/kaggle/working/outputs/` during training:
```
outputs/
├── efficientnet_b3_best.pt      # Best MH model checkpoint
├── crnn_best.pt                 # Best CRNN checkpoint
├── eda_samples.png              # Sample training images grid
├── eda_char_dist.png            # Character frequency & per-position distribution
├── augmentation_samples.png     # Augmented versions of one image (12 variants)
├── training_curves.png          # Loss / CER / Acc vs. epoch (MH model)
├── mh_val_analysis.png          # Per-position accuracy + edit-distance distribution
├── mh_val_predictions.png       # Validation image grid with predictions
├── crnn_curves.png              # CRNN training history
└── submission_Parambrata_Sinha_25112071.csv
```

---

## 10. How to Reproduce

### Requirements

```bash
pip install timm albumentations torch torchvision tqdm scikit-learn pandas matplotlib pillow
```

### Environment

Tested on:
- GPU: Tesla P100-PCIE-16GB (CUDA sm_60)
- PyTorch ≥ 2.0 (cu121 build)
- Python 3.10+

> **Note:** The notebook uses `torch.amp.autocast()` with `dtype=torch.float16`. The P100 does not support `bfloat16`, so `float16` is explicitly specified. On newer hardware (A100, H100) you can switch to `bfloat16` for improved numerical stability.

### Steps

1. Mount the dataset from the Google Drive link provided in the problem statement to `/kaggle/input/datasets/parambratasinha/cig-ai-ml/cig_ps/`
2. Open `best-cig.ipynb` in a Kaggle notebook with GPU accelerator enabled
3. Run all cells top-to-bottom — the notebook is fully self-contained
4. The submission CSV is written to `/kaggle/working/outputs/submission_Parambrata_Sinha_25112071.csv`

The notebook includes a GPU compatibility check at startup to verify that the detected CUDA compute capability is supported by the installed PyTorch build, preventing silent failures from mixed CUDA versions.

### Reproducibility

All randomness is controlled via:
```python
SEED = 42
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 11. Further Improvements

The following ideas were identified during development but not included in the final submission due to time/compute constraints. Each has a realistic expected gain:

| Idea | Expected Gain | Notes |
|---|---|---|
| Larger backbone (`efficientnet_b4`, `convnext_base`) | +1–3% seq acc | Higher capacity, diminishing returns after b3 |
| 5-fold CV + ensemble all fold checkpoints | +2–4% seq acc | Most reliable gain; each fold sees different training data |
| Pseudo-labelling high-confidence test predictions | +1–2% seq acc | Use predictions where all 6 characters have >0.99 confidence as additional training data |
| More TTA passes (5–10 varied augmentations) | +0.5–1% | Diminishing returns beyond ~5 passes |
| MixUp / CutMix on image pairs | +0.5–1% | Encourages smoother decision boundaries |
| Beam-search CTC decoding | CRNN-specific | Replaces greedy CTC decode with a more accurate beam search |
| Focal loss for hard examples | Variable | Useful if per-class accuracy analysis reveals persistently hard characters |
| Vision Transformer (ViT-Small) backbone | +1–3% | Attention mechanism better captures long-range character interactions |

---

## Submission Format

```csv
image,prediction
test-0.png,AXU323
test-1.png,BT91KD
test-2.png,QWER12
```

All predictions are validated to:
- Have filenames starting with `test-`
- Be exactly 6 characters long
- Contain only characters from the 31-character vocabulary (`23456789ABCDEFGHJKMNPQRSTUVWXYZ`)

---

*Solution by Parambrata Sinha (25112071) — CIG AI/ML Competition*