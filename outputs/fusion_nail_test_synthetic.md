# Synthetic fusion on the nail test set

Source: `outputs/fusion_nail_test_synthetic.csv`.  
155 patients from the original nail test split (102 Normal, 53 anemic).

Models: boosted-anemia nail ResNet and the aligned synthetic palm network. Anemic nail crops are the pallor-shifted copies. Normal nail crops are the original photos. Palm frames come from `video_frame_selection` when they exist. **35 anemic IDs have no palm frames**, so fusion for those IDs is the nail vote alone. All 102 normals have both modalities, and 18 anemic IDs have both.

Each modality votes with `s = tanh(logit(p) - logit(0.50))`. Fusion is the quality-weighted average `S`. `|S| <= 0.2` is **uncertain**. Otherwise the sign of `S` is the call.

A modality **supports anemia** when `p >= 0.50`, and **normal** otherwise. The two modalities **disagree** when those hard votes differ.

## Fusion as a whole

Calls: **54 anemic**, **74 normal**, **27 uncertain**.

Metrics below are on the **128 certain calls**. The last row scores every uncertain call as wrong, so abstaining is not free.

| | Accuracy | Precision | Recall | Specificity | F1 | TN / FP / FN / TP |
|--|----------|-----------|--------|-------------|----|-------------------|
| Certain calls (128) | 0.961 | 0.926 | 0.980 | 0.948 | 0.952 | 73 / 4 / 1 / 50 |
| Uncertain counted wrong (155) | 0.794 | | | | | |

The single certain miss is ID091 (anemic, nail only, nail `p = 0.43`). The four certain false anemia calls are ID210, ID481, ID724, and ID868, all normals with both modalities.

Split by which modalities were present:

| Subset | Rows | Uncertain | Certain accuracy | Precision | Recall | F1 | TN / FP / FN / TP |
|--------|------|-----------|------------------|-----------|--------|----|-------------------|
| Both modalities | 120 | 26 | 0.957 | 0.810 | 1.000 | 0.895 | 73 / 4 / 0 / 17 |
| Nail only (all anemic) | 35 | 1 | 0.971 | 1.000 | 0.971 | 0.985 | 0 / 0 / 1 / 33 |

Recall on the certain both-modality rows is 1.0 because every anemic patient who had a palm frame and a certain call was found. Precision is lower there (0.81) because all four false anemia calls are in that group.

## Scoring every ID by the sign of S

Uncertain rows still have an `S`. This section ignores the `|S| <= 0.2` band and uses the same sign rule as a certain call: **S <= 0 is normal, S > 0 is anemic**. All 155 IDs get a label. The 27 uncertain rows become **2 anemic** and **25 normal**.

| Subset | N | Anemic / normal calls | Accuracy | Precision | Recall | Specificity | F1 | TN / FP / FN / TP |
|--------|---|------------------------|----------|-----------|--------|-------------|----|-------------------|
| All test IDs | 155 | 56 / 99 | 0.955 | 0.911 | 0.962 | 0.951 | 0.936 | 97 / 5 / 2 / 51 |
| Previously uncertain only | 27 | 2 / 25 | 0.926 | 0.500 | 0.500 | 0.960 | 0.500 | 24 / 1 / 1 / 1 |
| Modalities agree | 86 | 20 / 66 | 0.965 | 0.850 | 1.000 | 0.957 | 0.919 | 66 / 3 / 0 / 17 |
| Modalities disagree | 34 | 2 / 32 | 0.912 | 0.000 | 0.000 | 0.939 | 0.000 | 31 / 2 / 1 / 0 |

Agreement has no uncertain rows, so that line matches the certain-call table. On the 34 disagreements, forcing the sign calls 32 normal and 2 anemic. Thirty-one of the 33 true normals are called normal. The one true anemic disagreement has `S <= 0`, so it is called normal. That is why precision and recall on the disagreement rows are 0: there is no true positive, one false negative, and two false positives.

## When the modalities agree

86 of 120 paired IDs agree. None of those 86 are uncertain.

| Agreed vote | True normal | True anemic |
|-------------|-------------|-------------|
| Both normal | 66 | 0 |
| Both anemic | 3 | 17 |

Certain-call metrics on agreement: accuracy **0.965**, precision **0.850**, recall **1.000**, specificity **0.957**, F1 **0.919**. Confusion: TN 66, FP 3, FN 0, TP 17. The three false anemia calls are cases where both models voted anemic on a normal patient.

## When the modalities disagree

34 paired IDs disagree. Every one of them is the same pattern: **nail supports normal, palm supports anemic**.

| True label | Count | Nail correct | Palm correct |
|------------|-------|--------------|--------------|
| Normal | 33 | 33 | 0 |
| Anemic | 1 | 0 | 1 |

On these 34 disagreements the nail vote matches the label 33 times and the palm vote matches it once.

Fusion does not pick a side most of the time:

| Fusion result | Count |
|---------------|-------|
| Uncertain | 26 |
| Follows nail | 7 |
| Follows palm | 1 |

The 8 certain calls inside the disagreement set: TN 7, FP 1, FN 0, TP 0. Accuracy **0.875**. Precision and recall are 0 because that set has no true anemia call: the one true anemic disagreement stayed uncertain, and the one certain anemia call was a false positive (fusion followed the palm). Counting the 26 abstentions as wrong drops accuracy on the 34 disagreements to **0.206**.
