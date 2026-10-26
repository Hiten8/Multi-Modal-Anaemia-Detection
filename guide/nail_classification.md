# Nail image classification

This note describes the **nail branch**: what we tried, which model we kept, how it is built, and the test numbers we use for fusion.

The task is **binary screening**: **Normal** vs **Anemia** (Spanish labels **Leve** + **Moderada**). Images come from the UNAS-style nail dataset. We do **not** use lab hemoglobin as an input at inference.

---

## Data and split

- **Input images:** manual nail crops (`unas_cropped`), not full-frame finger photos. Crops are RGB, resized to **224×224**, then **ImageNet** mean/std normalize.
- **Folders:** `Normal/`, `Leve/`, `Moderada/`. Binary label: `y = 0` if Normal, else `1`.
- **Local crop set used for the quality CSVs:** 775 images (509 Normal, 194 Leve, 72 Moderada).
- **Split (same everywhere):** stratified on the **3-class** label, `random_state=42`.
  - 80% train / 20% test
  - 15% of the remaining train held out as **validation**
  - Quality-pipeline CSVs: **527 train**, **155 test**

Colab sometimes used a slightly larger Drive `unas` / `unas_cropped` dump; architecture and protocol are the same.

**Preprocess (CNN path):** crop → RGB → resize 224 → (train only) horizontal flip and ±10° rotation → tensor → ImageNet normalize. **No CLAHE** on the CNN path.

---

## What we tried (and did not keep)

| Experiment | Outcome |
|------------|---------|
| 3-class ResNet-50 (Normal / Leve / Moderada) | Near-chance macro-F1; mostly predicts Normal; Moderada almost unused |
| Random Forest on handcrafted colour/texture | Collapses to always-Normal on test |
| Auto ROI (HSV finger + top crop) | High recall, many false positives; not better than a clean manual crop |
| Full-frame frozen ResNet-50 | AUC ~0.73; similar to crops, not a clear win |
| EfficientNet-B3 on crops | Tied or worse AUC than ResNet-50 |
| Unfreeze `layer4` and fine-tune | Usually worse val/test than a **frozen** backbone |

**Conclusion:** stop hunting backbones. The nail CNN is a **weak screen** (ranking is OK; precision at high recall is not). Value is in **fusion** with other non-invasive branches, not another classifier.

---

## Model we use

**Name:** frozen **ResNet-50**, ImageNet-pretrained (`IMAGENET1K_V2`), trained on **manual crops**.

**Checkpoint (Drive):**  
`MyDrive/anemia/outputs/resnet50_binary_crop_frozen_best.pt`

**Crops (Drive):**  
`MyDrive/unas_cropped/`

### Architecture

ResNet-50 is a residual CNN. We keep the **ImageNet convolutional body** and replace the classifier:

1. **Stem + residual stages `layer1`–`layer4`**  
   Standard ResNet-50 (bottleneck blocks). Output is a **2048-D** pooled vector.
2. **All of those weights are frozen** (`requires_grad = False`).  
   They stay ImageNet features; we do not update them.
3. **Head:** `Linear(2048 → 2)`  
   Two logits: class 0 = Normal, class 1 = Anemia.  
   **Only this layer is trained.** During a training step the backbone stays in eval mode and the head is in train mode.
4. **Probability:**  
   \( p = \mathrm{softmax}(z)_1 \)  
   i.e. P(Anemia). This is **not** a sigmoid head (palm uses sigmoid; nails use 2-class softmax).

### Training recipe

- **Loss:** weighted cross-entropy (inverse class frequency on the train split) so Anemia is not ignored.
- **Optimiser:** Adam on **`fc` only**, learning rate \(10^{-3}\).
- **Epochs:** 15; keep the epoch with best **validation anemia F1**.
- **Batch size:** 32.

Inference: load the `.pt` into the same architecture (`resnet50(weights=None)`, replace `fc`, `load_state_dict`). Same 224 + ImageNet normalize, **no** flip/rotation.

---

## How a prediction is made

1. Load crop → 224×224 → normalize.
2. Forward ResNet → two logits → **\(p\)**.
3. Compare \(p\) to a threshold **\(t\)**:  
   **Anemia** if \( p \ge t \), else **Normal**.

**AUC does not depend on \(t\)** (it only ranks). Accuracy / precision / recall **do**. We report three operating points used in the quality CSVs: **0.45, 0.50, 0.60**.

- **\(t = 0.50\)** — default `argmax` / softmax cut; used as the usual nail packet threshold.
- **\(t = 0.45\)** — more anemia calls (higher recall, more false positives).
- **\(t = 0.60\)** — fewer anemia calls (higher precision, more missed anemia).

---

## Test results (this repo’s quality test CSV)

Source: `outputs/nail_quality_test.csv`, **n = 155**, anemia prevalence ≈ **34%**.  
**ROC-AUC ≈ 0.75** (same band as the paper’s DenseNet-169 on this dataset family, ~0.74).

Confusion matrices are **\[\[TN, FP\], \[FN, TP\]\]** with Anemia as the positive class.

| Threshold \(t\) | Accuracy | Precision | Recall | F1 | Confusion |
|-----------------|----------|-----------|--------|-----|-----------|
| **0.45** | 0.66 | 0.50 | **0.83** | 0.62 | 58 / 44 / 9 / 44 |
| **0.50** | 0.68 | 0.53 | 0.68 | 0.60 | 70 / 32 / 17 / 36 |
| **0.60** | **0.74** | **0.68** | 0.47 | 0.56 | 90 / 12 / 28 / 25 |

Notes:

- **Accuracy is a poor headline** here: a dummy “always Normal” already scores about **66%**.
- Colab runs of the same recipe sometimes showed **higher recall at \(t = 0.50\)** (~0.85) with **precision ~0.45**. AUC stayed ~**0.75**. Treat threshold metrics as **run-dependent**; ranking is the stable result.
- The branch is a **high-recall / modest-precision** screen at low \(t\), and a **cleaner but miss-prone** vote at high \(t\).

**Recommended packet \(t\):** **0.50** unless fusion experiments explicitly compare 0.45 / 0.60.

---

## Inference for fusion

You do **not** need to retrain. Either:

- reuse `p` (and `q_mean`) from `outputs/nail_quality_test_with_q.csv` for **test IDs**, or  
- Colab notebook `notebooks/nail_binary_crop_resnet_threshold_colab.ipynb`: mount Drive, load the `.pt`, same split seed **42**, write probabilities.

Join fusion on filename stem **`ID086`**, same ID as palm. Only **nail test** IDs are valid held-out nail probabilities (do not use train-split \(p\) as if it were test).
