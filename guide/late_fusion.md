# Late fusion

Late fusion means: **each modality runs its own model**, then we combine **small packets of numbers** — not pixels. Nail and palm can be trained and swapped independently. If a person has only nails, fusion still runs; we do not pad a dummy palm image through a joint network.

Hemoglobin from a blood test is **not** a fusion input. It may label or rank data for training; the deployed story is non-invasive.

---

## Why packets, not one CNN on all images

A model that concatenates nail + palm tensors needs **every** view at train and test. Missing palm then requires fake inputs or a second network.

Packets fix that:

```text
nail crop  → nail ResNet + q  → packet_nail
palm frames → palm CNN + q   → packet_palm   (omit if unavailable)

available packets → fuse → S, label, per-modality support
```

Adding a third modality is another adapter that fills the **same schema**.

---

## Packet schema

JSON array (or `{ "packets": [ ... ] }`). Example: `examples/fusion_packets.json`.

| Field | Type | Meaning |
|--------|------|---------|
| `modality_name` | string | `"nail"`, `"palm"`, … |
| `available` | bool | Was this sensor used? If false, the rest is **ignored** |
| `p` | \([0,1]\) | P(anemia) from that branch |
| `t` | \([0,1]\) | Operating threshold for **that** branch |
| `pred` | 0/1 | Hard call (optional; fusion uses \(p\) vs \(t\) for “supports”) |
| `q` | \([0,1]\) | Capture quality (nail: **q_mean**; palm: **1** until palm stats exist) |

**Do not** send \(p = 0.5\), \(q = 0\) as a fake “missing” vote. That can still affect the denominator. Omit the packet or set `available: false`.

**Nail:** \(p = \mathrm{softmax}\) anemia class, typical \(t = 0.50\), \(q =\) q_mean.  
**Palm:** patient-level mean of frame sigmoid probabilities, default \(t = 0.50\), \(q = 1\) for now.

---

## What we combine (and what we dropped)

We need:

- **Direction:** does this branch support Normal or Anemia **relative to its own \(t\)**?
- **Strength:** how far is \(p\) from \(t\) (in a bounded, smooth way)?
- **Listen-less:** bad photo (\(q\) small).

We **do not** use a separate margin \(m = \lvert p-t\rvert/\max(t,1-t)\) in the weight. After we chose **tanh-vs-\(t\)** for the signed score \(s\), distance to the cutoff is **already inside \(s\)**. Multiplying by another \(m\) would count certainty twice, and linear margin **disagrees** with tanh when \(t \neq 0.5\).

---

## Signed score \(s\) (tanh)

For each **available** modality:

\[
s = \tanh\bigl(k \cdot (\mathrm{logit}(p) - \mathrm{logit}(t))\bigr)
\]

\[
\mathrm{logit}(x) = \log\frac{x}{1-x}
\]

\(p\) and \(t\) are clipped slightly off \(0\) and \(1\) before logit so \(\log 0\) never happens.

| \(p\) vs \(t\) | \(s\) |
|----------------|--------|
| \(p = t\) | \(0\) |
| \(p > t\) | \(> 0\) (pull **anemic**) |
| \(p < t\) | \(< 0\) (pull **normal**) |
| far from \(t\) | \(\to \pm 1\) |

**Why this, not \(s = 2p-1\):** \(2p-1\) is signed vs **0.5**, not vs **\(t\)**. If \(t = 0.6\) and \(p = 0.55\), \(2p-1 > 0\) (anemia lean) but the product would **not** call anemia. Tanh-logit is **zero at \(t\)**, so the vote matches the threshold we actually use.

**Why tanh, not raw logit:** \(\mathrm{logit}(p)-\mathrm{logit}(t)\) goes to \(\pm\infty\). One overconfident \(p \approx 1\) would wipe the other modality. \(\tanh\) keeps \(s \in (-1,1)\).

**Why not piecewise linear \((p-t)\) stretched to \(\pm 1\):** that is also valid; we preferred tanh because it is smooth and saturates extremes instead of treating the last 0.01 of probability as a full extra step.

**\(k\)** (CLI `--k`, default **1**) is steepness. Larger \(k\) makes \(s\) jump faster away from \(t\). Keep \(k\) **fixed** when you compare patients or talk about “\(S\) near 0.”

---

## Fused score \(S\)

\[
S = \frac{\sum_i q_i s_i}{\sum_i q_i}
\]

sum only over **available** modalities with \(\sum q_i > 0\).

This is a **quality-weighted mean of \(s\)**. Ratios of \(q\) matter, not the absolute scale of \(q\).

| Situation | What happens |
|-----------|----------------|
| One modality | \(S = s\) for that branch (\(q\) cancels). Quality is then a **retake** hint, not a change in \(S\). |
| Two modalities | Higher \(q\) listens more. |
| High \(q\), \(s \approx 0\) | Adds almost nothing to the numerator but **full \(q\)** to the denominator → **shrinks \(S\) toward 0** (panel looks unsure). |
| All \(q = 0\) or nothing available | Fusion should **error / abstain**, not invent \(S = 0\). |

**Decision (current rule):**

- \(S \le 0\) → **normal**
- \(S > 0\) → **anemic**

A **tie \(S = 0\)** is labelled normal. For screening that is the more dangerous default (false reassurance). A small band \(\lvert S\rvert \le \delta\) (e.g. **0.2**) should later be **uncertain**, not a hard call.

**\(S\) is not \(P(\text{anemia})\).** Do not report \(S = 0.2\) as “20% anemic.”

Hard AND/OR on `pred` bits are **other policies** on the same packets (high precision vs high recall). They are not this average.

---

## Per-modality print (why it matters)

Besides \(S\), each available branch should report:

- `modality_name`
- `p`
- **supports:** normal if \(p \le t\), anemic if \(p > t\)
- `q`
- `s`
- contribution \(s \cdot q\) (this row’s term in the numerator)

\(\lvert S\rvert\) small can mean three different things:

| Pattern | Meaning |
|---------|---------|
| All same support, small \(\lvert s\rvert\) | Everyone is near their cutoff |
| **Conflict** (nail anemic, palm normal) | Disagreement, not a weak agreement |
| One large \(\lvert s\cdot q\rvert\), one tiny | Loud branch is driving \(S\) |

Without those lines, \(S = +0.05\) looks like a weak anemia call. With them, it might be a **fight**.

---

## Late fusion vs training on all images together

| Joint CNN | Late fusion |
|-----------|-------------|
| Needs all modalities every time | 1, 2, or N packets |
| Retrain to add palm | Register a new packet source |
| Missing view is a special case | `available: false` |
| Shares pixels / features | Only shares the **schema** (\(p,q,t\)) |

Cost: fusion cannot fix a weak nail CNN by “seeing” palm pixels. It can only **reweight** votes. If both branches are high-FP, \(S\) stays high-FP unless \(q\) actually marks bad captures (on nails, that link is **weak**).

---

## Testing the framework with palm

Palm inference (their package): frames → EfficientNet-B0 sigmoid \(p\) → **mean per patient** at \(t = 0.5\).

Nail test IDs and palm **test** IDs share only about **26** people if splits were independent. That is enough to **exercise packets and \(S\)**, not enough for a strong “fusion wins” claim. Do not use nail **train** \(p\) on palm-test IDs (leakage).

---

## Implementation note

Designed CLI: `src/fusion/fuse.py` (`--json`, `--k`). Packet example: `examples/fusion_packets.json`. If `fuse.py` is missing from a checkout, the formulas in this guide are the spec to reimplement.
