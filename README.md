# Anemia Detection — Tabular (Hb/CBC) Branch

Baseline tabular branch of the Multi-Modal Anemia Detection project.
This trains and evaluates two models on Gender + Hb + CBC indices
(MCH, MCHC, MCV) to predict anemia (`Result`: 0 = No Anemia, 1 = Anemia):

1. **XGBoost** (primary model for this branch)
2. **Logistic Regression** (simple, interpretable baseline)

## ⚠️ Important data caveat — read before using these results in the paper

This dataset's `Result` column is not an independent clinical diagnosis —
it is **deterministically derived from Gender + Hemoglobin** using a fixed
threshold rule (this is confirmed by direct inspection, not a guess):

- Gender 1 (male): Hb < 13.5 → Anemia
- Gender 0 (female): Hb < 12.0 → Anemia

That's why both models score at or near 100% — there's no real prediction
task happening, just the model rediscovering a threshold rule already
baked into the labels. This is **not** a defensible result for peer review
and won't reflect real diagnostic difficulty. Two implications for the
project:

- Use this dataset only as a **coding/pipeline sanity check** (verifying
  your training loop, metrics, and file structure work correctly).
- Reserve the **Asare et al. multimodal dataset** as the real Hb/CBC
  tabular source for the paper, since its Hb values there come from
  actual lab measurements paired with imaging, not a synthetic rule.

## Folder structure

```
anemia_detection/
├── data/
│   └── anemia.csv                     # your dataset
├── src/
│   ├── utils.py                       # shared preprocessing + evaluation
│   ├── train_xgboost.py               # XGBoost training script
│   └── train_logistic_regression.py   # Logistic Regression training script
├── outputs/
│   ├── models/                        # saved .joblib model files (created on run)
│   └── results/                       # saved metrics .json files (created on run)
├── requirements.txt
└── README.md
```

## Setup instructions

### 1. Create the folder structure (if starting from scratch)

```bash
mkdir -p anemia_detection/data
mkdir -p anemia_detection/src
mkdir -p anemia_detection/outputs/models
mkdir -p anemia_detection/outputs/results
```

Place your `anemia.csv` inside `anemia_detection/data/`.
Place `utils.py`, `train_xgboost.py`, and `train_logistic_regression.py`
inside `anemia_detection/src/`.

### 2. Create a virtual environment (recommended)

```bash
cd anemia_detection
python3 -m venv venv

# activate it:
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you're on Google Colab instead of a local machine, skip the venv step
and just run `!pip install -r requirements.txt` in a cell, then upload
`anemia.csv` to a `data/` folder in your Colab session (or mount Drive).

## Running the code

From inside the `anemia_detection/` folder:

```bash
python src/train_xgboost.py
python src/train_logistic_regression.py
```

Each script will:
1. Load and validate `data/anemia.csv`
2. Do a stratified 80/20 train-test split (same random seed in both
   scripts, so results are directly comparable)
3. Train the model
4. Print accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix
5. Print feature importances (XGBoost) or standardized coefficients
   (Logistic Regression) — useful for the paper's discussion section
6. Save the trained model to `outputs/models/`
7. Save the metrics as JSON to `outputs/results/`

## Next steps for the project

- Once you swap in the Asare et al. dataset's genuine Hb/CBC values,
  re-run both scripts unchanged (same column names, same target
  encoding) as your real tabular-branch baseline.
- Add a quality/reliability signal for tabular data (e.g. a missing-value
  flag) as described in the fusion design — this doesn't affect XGBoost/
  Logistic Regression training itself, but will matter once this branch's
  output feeds into the attention-based fusion gate alongside the visual
  branches.
- Consider 5-fold stratified cross-validation instead of a single split
  once you're working with the real dataset, to get a more robust
  estimate for the paper (a single 80/20 split can be noisy on ~900
  patients).
