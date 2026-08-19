# SA-Co/Gold: both fine-tuned SAM3 checkpoints vs. stock SAM3

Out-of-domain evaluation of the two SAM3 fine-tunes this project produced, on Meta's
**SA-Co/Gold** promptable-concept-segmentation benchmark — all 7 subsets, 168,202
(image, noun-phrase) pairs, for all three models. Neither fine-tune saw any of this data in
training, so this is the conservative generalization measure for both.

| checkpoint | trained on | steps |
|---|---|--:|
| `reasonseg` | SA-1B referring expressions from the [data engine](README.md) — 18,146 positives + 181,460 verified hard negatives | 8,144 (1 epoch) |
| `mmr` | MMR train — 154,124 (image, question) samples, **`include_negatives: false`** ([`mmr.md`](mmr.md)) | 45,618 (1 epoch) |

**Headline.** The reasonseg fine-tune costs **−6.30 cgF1** (54.00 → 47.70). The MMR fine-tune
**collapses to 9.91** (−44.09). In both cases the loss is *presence classification*, not
segmentation: mask recall is 0.615 / 0.627 / 0.608 across the three models — statistically
indistinguishable — while image-level false-positive rate goes 0.022 → 0.060 → **0.847**. A
threshold sweep (§ below) confirms MMR's failure is lost *discriminability*, not a bad operating
point: no threshold makes it usable.

## Protocol

- **Data:** all 7 SA-Co/Gold subsets, scored against **3 independent annotators** with oracle
  (most-favourable) selection, instance-exhaustive queries only, as the benchmark specifies.
- **Metric:** **cgF1**, the official primary metric — classification-gated F1, jointly scoring
  detection and mask quality, gated on correctly classifying the concept present/absent. Reported
  alongside **IL_MCC** (image-level present/absent classification, class-imbalance robust) and
  **positive_micro_F1** (detection + mask quality on *positive pairs only*, i.e. with the presence
  decision taken out).
- **Inference:** `Sam3Processor`, text prompt only, bf16 autocast, masks at > 0.5. One image encode
  per picture, reused across that picture's phrases.
- **Scoring:** the repo's own `CGF1Evaluator` (`sam3/eval/cgf1_eval.py`), unmodified.

### The evaluator's threshold is 0.5, not 0.05

Worth stating prominently because it is undocumented and it changes how the numbers should be read.
`CGF1Evaluator.__init__` constructs `CGF1Eval` **without passing `threshold`**, so it silently uses
that class's default of **0.5**. Detections scoring below 0.5 are discarded before the metric sees
them, regardless of the confidence threshold used at inference (the SAM3 eval configs use 0.05).

Consequences:

- Every number here is a **score ≥ 0.5** number.
- Predicting at 0.05 vs 0.45 gives **bit-identical metrics**. Verified directly: filtering `mmr`'s
  `sa1b` predictions from 1,187,865 to 44,977 detections (3.8 %) leaves all metrics unchanged to
  6 decimal places (cgF1 0.141299 either way). `gold_eval.py --conf` exposes this; the two largest
  `mmr` subsets were run at 0.45, cutting ~19 h of compute to 52 min for identical results.
- `mmr` emits ~90 masks/pair at 0.05 but only **2.63/pair survive 0.5** — against base's 2.24. Its
  false positives are *confident*, not low-score noise. This matters for the diagnosis below.

### Harness validation

`gold_eval.py` had to be rebuilt for this run (paths had moved; `benchmark_refexp.build_model` was
gone). One trap worth recording: `model_builder._load_checkpoint` keeps only state-dict keys
containing `detector.`, which matches **zero** tensors in these trainer checkpoints — it would have
silently evaluated untrained weights and reported plausible numbers. The replacement loader asserts
an exact 1134/1134 tensor match and exits on mismatch.

Stock SAM3 was then run through the rebuilt harness on all 7 subsets and compared to the figures
Meta publishes in `sam3/scripts/eval/gold/README.md`:

| subset | measured here | Meta published | Δ |
|---|--:|--:|--:|
| Captioner MetaCLIP | 47.27 | 47.26 | +0.01 |
| Captioner SA-1B | 53.60 | 53.69 | −0.09 |
| Crowded | 60.88 | 61.08 | −0.20 |
| FG food | 53.22 | 53.41 | −0.19 |
| FG sport | 65.59 | 65.52 | +0.07 |
| Attributes | 55.15 | 54.93 | +0.22 |
| Wiki common | 42.25 | 42.53 | −0.28 |
| **average** | **54.00** | **54.06** | **−0.06** |

Mean absolute deviation 0.15 cgF1; IL_MCC matches to two decimals on all seven.

## Results — cgF1

| model | Cap. MetaCLIP | Cap. SA-1B | Crowded | FG food | FG sport | Attributes | Wiki common | **avg** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SAM 3 (stock) | 47.27 | 53.60 | 60.88 | 53.22 | 65.59 | 55.15 | 42.25 | **54.00** |
| `reasonseg` | 39.75 | 45.26 | 48.94 | 49.64 | 64.32 | 52.62 | 33.38 | **47.70** |
| `mmr` | 8.75 | 14.13 | 12.23 | 8.72 | 16.45 | 5.58 | 3.53 | **9.91** |
| Δ `reasonseg` | −7.52 | −8.34 | −11.94 | −3.58 | −1.27 | −2.53 | −8.87 | **−6.30** |
| Δ `mmr` | −38.52 | −39.47 | −48.65 | −44.50 | −49.14 | −49.57 | −38.72 | **−44.09** |

## Results — IL_MCC (present/absent classification)

| model | Cap. MetaCLIP | Cap. SA-1B | Crowded | FG food | FG sport | Attributes | Wiki common | avg |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SAM 3 (stock) | 0.806 | 0.858 | 0.899 | 0.790 | 0.889 | 0.765 | 0.696 | **0.815** |
| `reasonseg` | 0.705 | 0.764 | 0.769 | 0.765 | 0.878 | 0.745 | 0.575 | **0.743** |
| `mmr` | 0.164 | 0.255 | 0.211 | 0.135 | 0.257 | 0.090 | 0.063 | **0.168** |

## Results — positive_micro_F1 (quality on positive pairs only)

| model | Cap. MetaCLIP | Cap. SA-1B | Crowded | FG food | FG sport | Attributes | Wiki common | avg |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SAM 3 (stock) | 58.64 | 62.48 | 67.70 | 67.37 | 73.78 | 72.10 | 60.67 | **66.10** |
| `reasonseg` | 56.42 | 59.26 | 63.62 | 64.91 | 73.24 | 70.62 | 58.05 | **63.73** |
| `mmr` | 53.40 | 55.37 | 57.98 | 64.73 | 64.10 | 62.05 | 56.50 | **59.16** |

## The diagnostic that explains both fine-tunes

Means over all 7 subsets:

| metric | SAM 3 | `reasonseg` | `mmr` |
|---|--:|--:|--:|
| cgF1 | 0.540 | 0.477 | 0.099 |
| precision | 0.663 | 0.555 | 0.186 |
| **recall** | 0.615 | 0.627 | **0.608** |
| positive_micro_F1 | 0.661 | 0.637 | 0.592 |
| IL_precision | 0.899 | 0.771 | 0.254 |
| **IL_recall** | 0.797 | 0.816 | **0.991** |
| **IL_FPR** | 0.022 | 0.060 | **0.847** |
| IL_MCC | 0.815 | 0.743 | 0.168 |

Read **recall** against **IL_FPR**. All three models find what is there at the same rate
(0.615 / 0.627 / 0.608). What separates them is how often they claim something is there when it is
not: **2 % → 6 % → 85 %**. cgF1 collapses by 44 points while mask quality on positive pairs falls
only 7 (66.10 → 59.16).

## Threshold sweep: is `mmr` merely miscalibrated?

Re-scored existing `sa1b` predictions at a range of score thresholds (CPU only, no GPU — the
evaluator's threshold is the only knob that moves). This separates "cannot do the task" from
"wrong operating point".

| threshold | `mmr` cgF1 | `mmr` IL_MCC | `mmr` IL_FPR | `mmr` IL_recall | | SAM 3 cgF1 | SAM 3 IL_MCC | SAM 3 IL_FPR | SAM 3 IL_recall |
|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|
| 0.05 | 0.07 | 0.011 | 0.9998 | 1.000 | | 10.22 | 0.790 | 0.254 | 0.994 |
| 0.20 | 0.38 | 0.016 | 0.9996 | 1.000 | | 35.62 | 0.868 | 0.122 | 0.976 |
| 0.35 | 3.71 | 0.083 | 0.987 | 0.999 | | 48.68 | **0.874** | 0.081 | 0.953 |
| **0.50** *(evaluator default)* | 14.13 | 0.255 | 0.861 | 0.987 | | **53.60** | 0.858 | 0.055 | 0.918 |
| **0.65** | **19.59** | 0.399 | 0.544 | 0.895 | | 50.92 | 0.803 | 0.034 | 0.841 |
| 0.80 | 12.83 | **0.409** | 0.211 | 0.620 | | 35.75 | 0.671 | 0.016 | 0.663 |
| 0.90 | 5.18 | 0.327 | 0.058 | 0.319 | | 13.03 | 0.464 | 0.005 | 0.374 |

**1. Tuning helps `mmr` a little, and does not rescue it.** Its optimum is 0.65:
**14.13 → 19.59 cgF1 (+5.46)**, closing **14 %** of the 39.5-point gap to stock SAM3. The remaining
86 % is not a threshold artifact.

**2. The protocol is fair.** Stock SAM3's cgF1 peaks at exactly 0.5 — the evaluator's default. The
hidden threshold is well calibrated for the model the benchmark was built around, and it is within
5.5 cgF1 of the best `mmr` can achieve anywhere.

**3. `mmr` has no usable operating point.** A threshold only slides a model along its own ROC curve.
To bring `mmr`'s IL_FPR down to stock SAM3's 0.055 you must go to 0.9, where IL_recall falls to
**0.319** against base's 0.918. Suppress the false positives and it stops finding anything.

**4. The decisive number is IL_MCC, not cgF1.** Best achievable: `mmr` **0.409**, stock SAM3
**0.874**. The presence head has not been shifted off its optimum — it has become roughly half as
*informative*, and no threshold restores discriminability.

## Reading the reasonseg result

**1. The regression is presence classification, not segmentation.** `positive_micro_F1` falls only
**66.10 → 63.73 (−2.4)** while cgF1 falls **54.00 → 47.70 (−6.3)** and IL_MCC **0.815 → 0.743**. On
pairs where the concept is genuinely present, the fine-tune segments nearly as well as stock SAM3.

**2. The per-subset spread is large — −1.3 to −11.9 — and the mechanism differs by subset.** The
"−0.085 cgF1, over-triggering" line in [`README.md`](README.md) was measured on `sa1b` alone and
does not generalize:

| subset | Δ cgF1 | IL_FPR (base → ft) | IL_recall (base → ft) | mechanism |
|---|--:|--:|--:|---|
| FG sport | −1.27 | 0.006 → 0.022 | 0.851 → 0.909 | essentially free |
| Attributes | −2.53 | 0.048 → 0.081 | 0.823 → 0.880 | essentially free |
| FG food | −3.58 | 0.006 → 0.018 | 0.711 → 0.761 | mild, both directions |
| Captioner MetaCLIP | −7.52 | 0.014 → 0.057 | 0.761 → 0.782 | mask/detection quality |
| Captioner SA-1B | −8.34 | **0.055 → 0.153** | 0.918 → 0.913 | **over-triggers** (README's case) |
| Wiki common | −8.87 | 0.004 → 0.013 | **0.605 → 0.591** | recall-limited, not FPR |
| Crowded | −11.94 | 0.019 → 0.074 | 0.911 → 0.879 | instance-exhaustive counting |

Over-triggering dominates on **`sa1b` only** (FPR nearly triples). On `wiki_common` — the
second-worst subset — FPR is a negligible 0.013 and the loss is recall. On `crowded` neither
explains it: the model stops enumerating every instance.

**3. The damage tracks instances-per-phrase.** `crowded` (most instances per phrase) is worst;
fine-grained single-object domains (`fg_sport`, `attributes`) are nearly untouched. The engine mines
*single-referent* expressions ("the phone held by the woman in red"), so the fine-tune specializes
away from exhaustive multi-instance enumeration — the same trade the in-domain eval shows from the
other side (complex-referring IoU 0.427 → 0.707, multi-instance 0.996 → 0.899).

## Reading the mmr result

**The presence head collapsed; the mask head did not.**

| subset | cgF1 | IL_FPR | IL_recall | IL_MCC | positive_micro_F1 (base) |
|---|--:|--:|--:|--:|--:|
| Attributes | 5.58 | 0.963 | 0.999 | 0.090 | 62.05 (72.10) |
| FG food | 8.72 | 0.851 | 0.999 | 0.135 | 64.73 (67.37) |
| Captioner MetaCLIP | 8.75 | 0.862 | 0.991 | 0.164 | 53.40 (58.64) |
| Crowded | 12.23 | 0.795 | 0.980 | 0.211 | 57.98 (67.70) |
| Captioner SA-1B | 14.13 | 0.861 | 0.987 | 0.255 | 55.37 (62.48) |
| FG sport | 16.45 | 0.717 | 0.994 | 0.257 | 64.10 (73.78) |
| Wiki common | 3.53 | 0.879 | 0.989 | 0.063 | 56.50 (60.67) |

`IL_recall` is **0.980–0.999 on every subset**: the model answers "present" to essentially every
phrase. IL_MCC ≈ 0.17 is barely above chance. cgF1 then tracks how often that blanket "yes" happens
to be right, which is why `wiki_common` — where most phrases genuinely are absent — is worst at 3.53.

Meanwhile `positive_micro_F1` averages 59.16 against base's 66.10, and mean mask recall (0.608)
matches base's (0.615). On `fg_food` the mask gap is 2.6 points while cgF1 falls 44.5.

**Root cause.** MMR has no negative pairs — every question has an answer with masks — and the run
set `include_negatives: false`. The model was never given a reason to learn "not here". SA-Co/Gold
is roughly half negative pairs, so the presence gate cgF1 depends on is effectively untrained. This
is exactly the risk [`mmr.md`](mmr.md) §11 flagged as unmeasured ("we did not re-measure
SA-Co/Gold for this model"); it is real and severe.

**Why the threshold sweep matters here.** It rules out the benign explanation. This is not a
model whose scores drifted and need recentering — the ranking itself carries little information
(best IL_MCC 0.409 vs 0.874). Recovering it needs *training* with negatives, not inference-time
calibration. `reasonseg`, trained with 181,460 verified negatives, keeps IL_MCC at 0.743; that
contrast is the cleanest evidence in this project that the hard-negative half of the data engine is
doing real work.

## Coverage and limitations

- **Single run per configuration**, no seed variance. Differences below ~1 cgF1 (`fg_sport`'s
  −1.27) should not be over-read.
- **Threshold sweep is `sa1b` only.** The two `mmr` subsets run at `--conf 0.45` cannot be
  re-scored *below* 0.5 (nothing below was written). The sweep shows sub-0.5 is uniformly worse for
  both models, so nothing of interest lives there, but the option is gone for those subsets.
- **`reasonseg`'s `sa1b` and `metaclip` runs** predate a bit-identical optimisation to the
  mask-encoding path (on-device thresholding + batched RLE encode), verified to produce identical
  masks and RLE counts before adoption.
- **No retraining was attempted.** The obvious follow-up for `mmr` — mix negatives into MMR
  training — is untested here.

## Reproducing

```bash
PY=/workspace/envs/sam3/bin/python

# one subset, one checkpoint (omit --checkpoint for stock facebook/sam3)
$PY gold_eval.py --checkpoint <ckpt> --subset sa1b --device cuda:0 --out <out>.json

# all subsets for one model on one GPU; workers may share a model's queue (atomic mkdir claims)
bash gold_worker.sh <tag> <gpu-index> [checkpoint]

# one subset split across 3 GPUs, then merged and scored
bash run_gold_sharded.sh <tag> <checkpoint>

# threshold sweep on existing predictions (CPU only)
$PY gold_threshold_sweep.py --pred <preds>.json --subset sa1b --out <sweep>.json

# collate every metrics dump into the tables above
$PY gold_table.py base reasonseg mmr
```

Metric dumps: `/mnt/data0/ameen/gold_eval/{base,reasonseg,mmr}/<subset>_metrics.json`; sweeps in
`<tag>/sa1b_threshold_sweep.json`. Raw prediction JSONs are deleted after scoring.
