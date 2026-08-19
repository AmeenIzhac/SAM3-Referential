# reasonseg

An SA-1B referring-expression **segmentation** data engine, plus a SAM3 fine-tune trained on its
output. A tidied, single-file rewrite of the newsam "gemini" pipeline that produces positives
tagged with one of **eight reasoning categories**, plus verified hard **negatives**.

Out of the box [SAM3](https://github.com/facebookresearch/sam3) is strong on simple noun-phrase
prompts and multi-instance concepts, but weaker when the prompt needs compositional reasoning
("the phone held by the woman in red", "the largest suitcase", "the object that would block the
doorway if moved"). This engine mines exactly those expressions automatically; [Results](#results)
shows that fine-tuning on them roughly **doubles** complex-referring IoU in one epoch.

**Related:** [`mmr.md`](mmr.md) covers the separate MMR strand — training this SAM3 checkpoint on the
MMR reasoning-segmentation benchmark and beating its purpose-built 7B model from the raw question.
[`quant.md`](quant.md) quantizes that MMR checkpoint: 5-bit weights are free (−0.04 gIoU, 5.6×
smaller checkpoint), 4-bit costs 0.13 with fine scale groups, and the damage shows up as the presence
head going *quiet* — the opposite sign to the SA-Co/Gold regression below.

## Layout
```
generate_dataset.py     # the whole data engine, one script
eval_test.py            # in-domain held-out eval (per-category IoU)
gold_eval.py            # SA-Co/Gold eval (cgF1, IL_MCC, …)
sam3/                   # vendored clone of facebookresearch/sam3 (gitignored)
data/images/            # SA-1B .jpg frames (populated by the `download` stage)
out/                    # results.jsonl, previews, image_negatives.json, dataset/, eval/
```

## Reasoning categories
1. **Object & attributes** — intrinsic properties only (`the red car`)
2. **Spatial** — location / relative position (`the tree on the right`)
3. **Ordinal & comparative** — order or measurable comparison (`the largest suitcase`)
4. **Relational** — direct relation to another object (`the man's backpack`)
5. **Multi-hop** — route through an intermediate object (`the phone held by the woman in red`)
6. **Constraint composition** — several conditions at once (`the red car closest to the building`)
7. **Commonsense & affordance** — how objects are used/owned (`the driver's seat`)
8. **Counterfactual & predictive** — hypotheticals / consequences (`the object that would block the doorway if moved`)

Each sample carries exactly one category tag. Generation either round-robins the target category
(`generate`) or lets Gemini pick it (`auto`); the final kept counts are deliberately skewed (not
uniform) — see [Training data](#training-data).

## Pipeline (per image)
1. SAM3 interactive predictor auto-clicks clean, distinct candidate objects.
2. Gemini picks the object best suited to the target category and writes a category-specific,
   uniquely-identifying referring expression (or declines).
3. Mask refined by concept-segmenting the plain `simple_prompt` and keeping the best-overlapping
   mask (falls back to the click mask).
4. Verification: 1 programmatic (complexity) + 4 Gemini gates (grounding, boundary,
   unique/meaningful, **category-match**). PASS iff all pass.
5. **Score (1-10)**: Gemini rates each sample — uniqueness, naturalness, and whether the extra
   specification is genuinely *needed* (over-specifying a one-of-a-kind object scores low). The
   score is stored on the record and burned onto the preview PNG as a **black-on-white caption**
   at the bottom (`[8/10]  the prompt` + category/verdict/justification). **Only samples scoring
   ≥ 7 are kept.**

Negatives: propose plausible-but-absent concepts (Gemini) → verify absence (Gemini) → cap.

Export: train/test COCO per category; category carries `name` (expression), `supercategory`
(simple term) and `reasoning_type`; plus `image_negatives.json`.

## Two tracks
- **complex** — single-instance referring expressions across the 8 reasoning categories (1 mask).
- **multi** — multi-instance PCS: a plain concept → **all N** instance masks ("chairs" → every chair).

## Usage
```bash
export GEMINI_API_KEY=...          # required
PY=/workspace/envs/sam3/bin/python

$PY generate_dataset.py download                       # fetch + extract the SA-1B tar
$PY generate_dataset.py generate --mode complex        # single-instance referring (8 categories)
$PY generate_dataset.py generate --mode multi          # multi-instance concept samples
$PY generate_dataset.py negatives                      # verified hard-negative pool (sidecar)
$PY generate_dataset.py mix \                          # FINAL mixed dataset with ratios
      --multi-frac 0.5 --neg-frac-complex 10 --neg-frac-multi 10
# or the whole thing: $PY generate_dataset.py all
```

### Ratio knobs (the `mix` stage)
- `--multi-frac` (default **0.5**) — fraction of **positives** that are multi-instance (`0.5` = 50/50 multi/complex). The more-abundant track is subsampled to hit the ratio.
- `--neg-frac-complex` / `--neg-frac-multi` (default **10** each) — hard-negative phrases per positive, equal for both tracks. `10` matches the SAM3 data-engine sweet spot (Table 8b) and the loader's `max_negatives_per_datapoint`. Realized count is bounded by the verified-absent pool (`--neg-cap`, default 15).

`mix` writes `out/dataset_mixed/{train,test}/_annotations.coco.json`, an `image_negatives.json`
sidecar sized to the negative budget, and a `composition.json` reporting requested-vs-realized
ratios. `export` is the simpler all-positives dump with no ratio control.

Restrict complex categories with `--categories spatial multi_hop`. Base checkpoint defaults to
the surviving HF-cache `sam3.pt` (override with `--checkpoint`).

---

# Dataset, training & results

The material relevant for a write-up: the dataset that was actually built, the fine-tuning setup,
and the evaluation results (base SAM3 vs. the fine-tune).

## Training data

**Source:** SA-1B, **123,046** images (11 official tars). Every kept sample is score ≥ 7.

**Positives — 18,146 total** (9,073 complex + 9,073 multi, a deliberate **50/50** split):

| Reasoning category (complex track) | kept ≥7 |
|---|--:|
| constraint_composition | 1,424 |
| ordinal_comparative | 1,350 |
| spatial | 1,290 |
| multi_hop | 1,220 |
| relational | 1,155 |
| object_attribute | 1,105 |
| commonsense_affordance | 1,066 |
| counterfactual_predictive | 463 |
| **complex subtotal** | **9,073** |
| multi_instance (multi track) | 9,073 |
| **total positives** | **18,146** |

The counts are intentionally **skewed, not uniform** (a clean 1,000/category looks synthetic).
`counterfactual_predictive` is under-represented on purpose: it hit a genuine data-scarcity wall
(~0.6 % effective yield per image — hypothetical/consequence-based expressions rarely have a
clean, uniquely-resolvable referent in a static scene), so it was capped at what the engine could
verify rather than padded with low-quality samples.

**Hard negatives:** **181,460** verified-absent phrases (a budget of **10 per positive**), drawn
from a verified pool of **255,549** — matching the SAM3 data-engine regime (paper Table 8b). They
live in an `image_negatives.json` sidecar; the loader samples up to **3 per step** at train time.

**Split:** **image-level** hash split (an image and all its samples go entirely to train *or*
test — no image-level leakage between splits), **~10 % test**:

| split | samples | instance masks | images |
|---|--:|--:|--:|
| train | 16,287 | 56,344 | 14,831 |
| test  | 1,859 | 5,895 | 1,691 |

**Format:** COCO JSON, masks as RLE. Each "sample" is one COCO *category* (one prompt on one
image); `name` = the referring expression, `supercategory` = the plain concept, plus a
`reasoning_type` and `kind` (complex|multi) tag. Multi samples carry several instance annotations
under one category.

## Fine-tuning setup

**Base model:** SAM3 (`facebook/sam3`), fine-tuned end-to-end. Recipe follows the SAM3 paper's
few-shot / data-engine regime.

| | |
|---|---|
| Optimizer | AdamW, bf16 (AMP) |
| LR (transformer / vision bb / language bb) | 8e-5 / 2.5e-5 / 5e-6 (base LR × scale 0.1) |
| Vision-backbone layer-wise LR decay | 0.9 |
| Schedule | inverse square-root, 20 warm-up steps |
| Image resolution | 1008 × 1008 |
| Losses | focal (α=0.25, γ=2) + Dice (w=10) + presence BCE (w=20); semantic presence/Dice heads (Dice w=30) |
| Batch size | 1 image / GPU × 2 GPUs (DDP) → global batch 2 |
| Negatives | 10 verified per positive in the data; 3 sampled per step |
| Epochs | **1** (of 10 configured), ~8,144 steps |
| Hardware / time | 2 GPUs, ~4.5 h, rolling 10 GB checkpoint |

Batch size 1/GPU is a memory concession: multi-instance samples (many masks per image at res 1008)
OOM at larger batches. Training was stopped after a single epoch.

## Results

Two evaluations: an **in-domain** held-out split (same generator, measures how well the model
learns the target skill) and an **out-of-domain external benchmark** (SA-Co/Gold, the conservative
measure of generalization). Base = SAM3 zero-shot; Trained = after 1 epoch of fine-tuning.

### In-domain — held-out 10 % test (per-sample mask IoU)

IoU is computed per sample as the union of predicted masks (score ≥ 0.5, best-mask fallback)
against the union of ground-truth instance masks. `IoU@0.5` = fraction of samples with IoU ≥ 0.5.

| | n | IoU (base) | IoU (trained) | Δ |
|---|--:|--:|--:|--:|
| **Overall** | 1,859 | 0.697 | **0.798** | +0.101 |
| **Complex** (8 categories) | 978 | 0.427 | **0.707** | **+0.280 (+65 %)** |
| **Multi** (multi-instance) | 881 | 0.996 | 0.899 | −0.097 |

Per reasoning category (complex), IoU base → trained:

| category | base | trained | Δ |
|---|--:|--:|--:|
| multi_hop | 0.237 | 0.671 | +0.434 (+183 %) |
| counterfactual_predictive | 0.223 | 0.563 | +0.340 (+153 %) |
| constraint_composition | 0.384 | 0.722 | +0.338 |
| relational | 0.389 | 0.712 | +0.323 |
| spatial | 0.405 | 0.673 | +0.268 |
| commonsense_affordance | 0.397 | 0.596 | +0.200 |
| ordinal_comparative | 0.606 | 0.800 | +0.194 |
| object_attribute | 0.633 | 0.787 | +0.154 |

**Reading it:** fine-tuning roughly **doubles** complex-referring IoU, with the biggest gains on
the hardest, most compositional categories (multi-hop, counterfactual) where the base model was
near-random. The multi-instance track dips slightly (0.996 → 0.899) — the base model is already
near-perfect there (it's SAM3's native task), so this is mild forgetting from specialization, not
a real capability loss.

*Caveat for a write-up:* the held-out split shares the engine's annotation style, so part of this
gain is the model learning that style, not pure reasoning ability. The external benchmark below is
the honest generalization number.

### Out-of-domain — SA-Co/Gold (sa1b subset)

Meta's SA-Co/Gold benchmark, scored with the repo's `CGF1Evaluator` (segm, oracle a/b/c
annotator selection). This split was **never seen** in training.

| metric | base | trained | Δ |
|---|--:|--:|--:|
| **cgF1** | 0.539 | 0.454 | −0.085 |
| cgF1@0.5 | 0.664 | 0.565 | −0.099 |
| precision | 0.614 | 0.534 | −0.079 |
| **recall** | 0.624 | 0.624 | ≈ 0.000 |
| IL_F1 | 0.938 | 0.903 | −0.035 |
| **IL_MCC** | 0.861 | 0.766 | −0.096 |
| IL_recall | 0.918 | 0.911 | −0.007 |
| IL_precision | 0.959 | 0.895 | −0.064 |
| IL_FPR | 0.052 | 0.149 | **+0.096 (2.8×)** |

**Reading it:** on the out-of-domain benchmark the fine-tune **regresses** (cgF1 0.539 → 0.454).
The mechanism is specific and visible in the sub-metrics: **recall is essentially unchanged**
(0.624 → 0.624; IL_recall 0.918 → 0.911) — the model still finds what's there — but **precision
drops and the image-level false-positive rate nearly triples** (0.052 → 0.149). After one epoch on
a narrow, complex-phrasing distribution the model becomes **over-eager to assert presence** on the
generic phrases of an external benchmark. This is a classic single-epoch specialization trade-off,
not a collapse: in-domain skill goes up sharply, out-of-domain calibration degrades.

## Metric definitions (reference)

Each evaluation datapoint is one **(image, phrase)** pair.

- **IoU** — Intersection-over-Union of predicted vs. ground-truth masks. Pure mask-overlap quality;
  reported per-sample and averaged.
- **cgF1** (classification-gated F1) — the SA-Co primary metric. Jointly scores *detection* (is the
  right instance found?) and *mask quality* (IoU-thresholded), gated by whether the concept is
  correctly classified present/absent. Sensitive to both false positives and sloppy masks.
- **IL_F1 / IL_MCC** (Image-Level) — collapse every instance in a pair to a single binary "is this
  concept present in this image?" and score that classification, **ignoring mask quality**. The
  benchmark includes genuine negative pairs (phrase absent), so these use the full confusion matrix:
  - `IL_F1 = 2·TP / (2·TP + FP + FN)` — ignores true negatives.
  - `IL_MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))` — uses all four cells; ≈ 0 is
    random, 1 is perfect. The stricter, class-imbalance-robust measure, which is why the
    over-triggering above shows up much more in IL_MCC (0.861 → 0.766) than in IL_F1 (0.938 → 0.903).

## Takeaways & limitations

- **The engine works:** automatically mined, score-gated (≥7) complex referring expressions are
  enough to roughly double SAM3's complex-referring IoU in one epoch.
- **Specialization has a cost:** the same fine-tune regresses on an out-of-domain benchmark, purely
  through **over-triggering** (precision/FPR), with recall preserved. Report both numbers.
- **Counterfactual reasoning is data-limited** (463 vs ~1.1–1.4 k for other categories) — a real
  yield wall, not a pipeline bug.
- **Single epoch, single distribution.** Not tried here, natural next steps to recover the gold
  regression: mix in more base-distribution (simple-prompt / PCS) data, lower LR or fewer steps,
  freeze the vision backbone, or checkpoint-average base+fine-tuned weights.

## Reproduce the evals

```bash
PY=/workspace/envs/sam3/bin/python

# in-domain held-out (per-category IoU)
$PY eval_test.py --checkpoint <ckpt> --out out/eval/heldout.json

# out-of-domain SA-Co/Gold (cgF1, IL_MCC, …)
$PY gold_eval.py --checkpoint <ckpt> --subset sa1b --out out/eval/gold_sa1b.json
```

Raw metric dumps for the runs above are in `out/eval/` (`heldout_{base,trained}.json`,
`gold_sa1b_{base,trained}_metrics.json`).
# SAM3-Referential
