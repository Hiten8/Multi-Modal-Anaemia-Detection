# Nail quality score

Quality **\(q \in [0,1]\)** is **not** a second anemia label. It answers: *does this crop look like the photos we trained on, as a capture?* Fusion can then **down-weight** blurry, tiny, or lamp-blown nails instead of treating every \(p\) as equally trustworthy.

We compute **picture statistics** on the **same 224×224 RGB crop** the CNN sees (before ImageNet normalize). Original width/height are taken **before** resize. Hue is **not** used as “bad quality” — pallor is the disease signal.

Reference implementation: `src/nail_branch/image_quality.py`. Train reference file: `outputs/nail_quality_train_stats.json`.

---

## Two kinds of signal

| Source | Examples | Role |
|--------|----------|------|
| **Picture** | brightness, contrast, sharpness, crop size | Capture / crop quality → **\(q\)** |
| **Model** | \(p\), entropy, \(\lvert p-t\rvert\) | How sure the CNN is (used in fusion as \(p\) and \(t\), not mixed into \(q\)) |

A **confident \(p\)** on a burned-out crop should still get **low \(q\)**. Picture \(q\) and model \(p\) are meant to stay separate.

---

## What we store while training

On the **train split only** (never test), we measure each crop and write a JSON of **per-attribute** summaries:

- **n**, **resize** (224), split seed  
- For each attribute: **mean**, **std** (population, `ddof=0`), **p05**, **p50**, **p95**

That file is the **reference distribution** of “what training nails looked like.”

Attributes (raw units):

| Attribute | What it is | Typical “bad” direction |
|-----------|------------|-------------------------|
| **brightness** | Mean luma \(Y = 0.299R+0.587G+0.114B\) | Too dark **or** too bright (\(\lvert z\rvert\)) |
| **contrast** | Std of \(Y\) | Too **low** (flat / milky) |
| **saturation** | Mean HSV \(S\) | Too **low** (grey / not a nail bed) |
| **highlight_frac** | Fraction of pixels with HSV \(V > 250\) | Too **high** (flash / clip) |
| **shadow_frac** | Fraction with \(Y < 10\) | Too **high** (missed nail / black box) |
| **sharpness** | Variance of Laplacian on grey | Too **low** (blur / upscaled speck) |
| **orig_w, orig_h, orig_area** | Size **before** resize | Too **small** (huge stretch to 224) |
| **aspect_ratio** | \(w/h\) | Far from train mean (\(\lvert z\rvert\)) |

Percentiles (p05/p95) are stored for inspection and possible gates. **Method D \(q\)** uses **mean and std**, not the percentiles, when turning a new image into \(z\)-scores.

Fit this JSON **once** on train crops. At inference you **do not** update it.

---

## What happens at inference

For a new crop:

1. Compute the **same raw attributes** (224×224 stats + original geometry).
2. For each attribute \(x\), compare to the **stored train** mean \(\mu\) and std \(\sigma\):

   \[
   z = \frac{x - \mu}{\sigma + \varepsilon}
   \]

   \(\varepsilon\) avoids divide-by-zero when an attribute is almost constant on train (e.g. almost no shadows).

3. Turn \(z\) into a one-sided **badness** \(b \ge 0\) so “unusual in a harmless direction” is not punished:

   | Mode | Formula | Used for |
   |------|---------|----------|
   | `abs` | \(b = \lvert z\rvert\) | brightness, aspect ratio |
   | `low_bad` | \(b = \max(0, -z)\) | contrast, saturation, sharpness, orig size |
   | `high_bad` | \(b = \max(0, z)\) | highlight_frac, shadow_frac |

   Example: **sharper** than train (\(z > 0\)) → \(b = 0\) for sharpness. **Blurrier** (\(z < 0\)) → \(b = \lvert z\rvert\). Extra contrast is allowed; mushy contrast is not.

4. Convert badness to a **per-attribute similarity** in \((0,1]\):

   \[
   s_i = \exp(-b_i^2 / 2)
   \]

   This is a Gaussian-like gate: \(b = 0\) → \(s = 1\) (looks like train); large \(b\) → \(s \to 0\).

So: **stored JSON = train \(\mu,\sigma\)**. **Inference = how many σ this image sits in a *bad* direction, then squash to \([0,1]\).** That *is* “how similar this capture is to what we trained on,” with an explicit notion of which side of the mean is harmful.

---

## From per-attribute scores to one \(q\)

We tried three aggregations of the \(s_i\):

\[
q_{\min} = \min_i s_i, \qquad
q_{\mathrm{prod}} = \prod_i s_i, \qquad
q_{\mathrm{mean}} = \mathrm{mean}_i\, s_i
\]

| Formula | Behaviour on this data |
|---------|-------------------------|
| **q_min** | Strict: one bad attribute floors the score. Median ~0.44. Often limited by **contrast**, then **saturation**. |
| **q_prod** | Very harsh (product of many numbers &lt; 1). Median ~0.16–0.27. Almost everything looks “bad.” |
| **q_mean** | Forgiving average. Median ~0.88. Test range roughly **0.51–1.0**. **This is the \(q\) we use for fusion.** |

**q_mean** is preferred because **q_min** / **q_prod** push almost every real crop toward “unusable,” which is useless as a fusion weight. On **manual** crops, truly awful captures are rare, so **q_mean** stays high; it still down-weights the **worst** tail.

---

## What \(q\) is and is not (from the test analysis)

On the **155** test crops, **q_mean** vs “CNN correct at \(t=0.50\)” had correlation ≈ **0.05** (not significant). Mean \(q\) for right vs wrong predictions differed by about **0.01**.

So:

- **High \(q\)** does **not** mean “the CNN is probably right.”
- **Low \(q\)** is only **weakly** linked to more errors (a few percentage points), with more **false positives** in the low-\(q\) tertile.
- Anemia **rate** is flat across \(q\) bins — quality is **not** a proxy for the label.
- Bins below 0.5 are **empty** on test; a hard gate “ignore if \(q < 0.5\)” never fires.

**Use in fusion:** treat \(q\) as a **soft weight** (listen less to odd captures), not as a diagnosis and not as a guarantee of correctness. Optional product rule: ask for a **retake** when \(q\) is in the bottom tail, rather than outputting a confident fused label.

---

## Files

| File | Contents |
|------|----------|
| `outputs/nail_quality_train_stats.json` | Train \(\mu,\sigma\), percentiles |
| `outputs/nail_quality_train.csv` | Raw attributes on train images |
| `outputs/nail_quality_test.csv` | Raw attributes + \(p\) + threshold preds |
| `outputs/nail_quality_*_with_q.csv` | Same plus `q_min`, `q_prod`, `q_mean` and per-attribute \(b_i,s_i\) |
| `src/nail_branch/compute_quality_q.py` | Adds \(q\) columns using the train JSON |

Palm does **not** yet ship this JSON. Until a palm train-stats file exists, fusion can set palm **\(q = 1\)** (no quality gate) without changing the packet schema.
