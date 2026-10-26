# Original fusion on the nail test set

Source: `outputs/fusion_nail_test_original.csv`.  
Same 155 patients as the synthetic run (102 Normal, 53 anemic).

Models: the original frozen nail ResNet (`resnet50_binary_crop_frozen_best.pt`) and the original palm EfficientNet (`efficientnet_b0_original_best.pth`). Nail crops are the original `unas_cropped` photos. Anemic palm frames are the non-synthetic frames. Normal palm frames are the real Normal folders. **All 155 IDs have both modalities.**

Each modality votes with `s = tanh(logit(p) - logit(0.50))`. Fusion is the quality-weighted average `S`. `|S| <= 0.2` is **uncertain**. Otherwise the sign of `S` is the call.

A modality **supports anemia** when `p >= 0.50`, and **normal** otherwise. The two modalities **disagree** when those hard votes differ. They can still agree and land in the uncertain band when both probabilities sit close to 0.50.

## Fusion as a whole

Calls: **37 anemic**, **70 normal**, **48 uncertain**.

Metrics below are on the **107 certain calls**. The last row scores every uncertain call as wrong.

| | Accuracy | Precision | Recall | Specificity | F1 | TN / FP / FN / TP |
|--|----------|-----------|--------|-------------|----|-------------------|
| Certain calls (107) | 0.813 | 0.676 | 0.758 | 0.838 | 0.714 | 62 / 12 / 8 / 25 |
| Uncertain counted wrong (155) | 0.561 | | | | | |

Of 53 anemic patients, 25 are certain true positives, 8 are certain misses, and the rest are uncertain. Of 102 normals, 62 are certain true negatives, 12 are certain false anemia calls, and the rest are uncertain.

## Scoring every ID by the sign of S

Uncertain rows still have an `S`. This section ignores the `|S| <= 0.2` band and uses the same sign rule as a certain call: **S <= 0 is normal, S > 0 is anemic**. All 155 IDs get a label. The 48 uncertain rows become **31 anemic** and **17 normal**.

| Subset | N | Anemic / normal calls | Accuracy | Precision | Recall | Specificity | F1 | TN / FP / FN / TP |
|--------|---|------------------------|----------|-----------|--------|-------------|----|-------------------|
| All test IDs | 155 | 68 / 87 | 0.723 | 0.574 | 0.736 | 0.716 | 0.645 | 73 / 29 / 14 / 39 |
| Previously uncertain only | 48 | 31 / 17 | 0.521 | 0.452 | 0.700 | 0.393 | 0.549 | 11 / 17 / 6 / 14 |
| Modalities agree | 85 | 32 / 53 | 0.871 | 0.750 | 0.889 | 0.862 | 0.814 | 50 / 8 / 3 / 24 |
| Modalities disagree | 70 | 36 / 34 | 0.543 | 0.417 | 0.577 | 0.523 | 0.484 | 23 / 21 / 11 / 15 |

Forcing a sign on the uncertain rows lowers accuracy from 0.813 on the 107 certain calls to **0.723** on all 155. Those 48 rows alone are about chance (accuracy 0.521): 17 of the 31 anemia calls in that band are false positives. On disagreements, the sign of `S` is right on 38 of 70 IDs (accuracy 0.543), with recall 0.577. That is weaker than the 29 certain disagreement calls (accuracy 0.621), because most of the band was sitting on the wrong side of zero.

## When the modalities agree

85 of 155 IDs agree. 7 of those 85 are still uncertain, because both votes are on the same side of 0.50 but `S` stays inside the 0.2 band.

| Agreed vote | True normal | True anemic |
|-------------|-------------|-------------|
| Both normal | 50 | 3 |
| Both anemic | 8 | 24 |

Certain-call metrics on agreement (78 calls): accuracy **0.885**, precision **0.778**, recall **0.875**, specificity **0.889**, F1 **0.824**. Confusion: TN 48, FP 6, FN 3, TP 21.

When both models are wrong together, fusion keeps that error: 8 normals both called anemic, and 3 anemic patients both called normal. Agreement is the stronger operating region. It is not automatically correct.

## When the modalities disagree

70 IDs disagree. On a disagreement exactly one modality matches the label.

| Pattern | True normal | True anemic |
|---------|-------------|-------------|
| Nail anemic, palm normal | 34 | 18 |
| Nail normal, palm anemic | 10 | 8 |

Across all 70 disagreements, the nail vote matches the label **28** times and the palm vote matches it **42** times.

Fusion mostly abstains:

| Fusion result | Count |
|---------------|-------|
| Uncertain | 41 |
| Follows palm | 21 |
| Follows nail | 8 |

On the 29 certain disagreement calls: accuracy **0.621**, precision **0.400**, recall **0.444**, specificity **0.700**, F1 **0.421**. Confusion: TN 14, FP 6, FN 5, TP 4. Fusion is right on 18 of those 29. Counting the 41 abstentions as wrong drops accuracy on the 70 disagreements to **0.257**.

The palm side is right more often than the nail side in this disagreement set, and the certain calls follow the palm more often (21 versus 8). The uncertain band absorbs most of the conflict rather than forcing a winner.
