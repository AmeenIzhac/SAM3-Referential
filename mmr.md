# SAM3 on MMR: multi-target, multi-granularity reasoning segmentation

Everything from the MMR strand of this project: head-to-heads against the benchmark's own model
(M²SA-7B) and two prior reasoning-segmentation systems (LISA-7B, PixelLM-7B), what happened when we trained SAM3 on MMR itself, the
training details, the inference cost, and an honest account of what went wrong on the way.

Post-training quantization of the checkpoint trained here is a separate document:
[`quant.md`](quant.md) — 8/6/5-bit weights are free, 4-bit costs 0.13 gIoU at 4.75 bits/weight, and
the model's precision bottleneck turns out to be the detection heads, not either backbone.

All baselines were re-run here rather than quoted, because the paper reports a *per-target* metric
that cannot be applied to SAM3's unordered mask output (§2) — every cross-model number below comes
from one harness scoring one identical ground truth.

**Headline.** MMR ships with M²SA-7B, a LLaVA-7B + SAM ViT-H model purpose-built for the benchmark.
Out of the box SAM3 scores **0.6 gIoU** on MMR — it cannot parse a reasoning question at all.
Trained on MMR itself, the same SAM3 checkpoint reaches **37.15 gIoU from the raw question**, above
M²SA-7B's **36.0** *teacher-forced* score (which is given the answer text naming the targets) and
**+11.9** over its honest question-only **25.2** — at **~42× lower latency per prediction**, using
**~2.4× less MMR supervision than the paper used** (§8). The gap between SAM3 and a reasoning-
segmentation model was language, not pixels, and in-distribution data closes almost all of it
without an LLM in the loop.

---

## 1. Task and benchmark

**MMR** (*Multi-target and Multi-granularity Reasoning Segmentation*, ICLR 2025,
[arXiv 2503.13881](https://arxiv.org/pdf/2503.13881), [jdg900/MMR](https://github.com/jdg900/MMR))
pairs an image with an *implicit reasoning question* — "What might be the purpose of the item
holding the flower?" — and asks for the pixel masks of the answer. Two properties make it harder
than ReasonSeg-style benchmarks:

- **Multi-target**: an answer can be several objects at once (5,186 of 8,194 val questions).
- **Multi-granularity**: targets can be whole objects **and their parts** (`cup:handle`), on
  PACO-LVIS masks.

We evaluate on the official **val** split throughout: **2,404 images / 8,194 (image, question)
pairs / 14,671 target masks**.

## 2. Metrics, and one measurement decision that matters

Per (image, question) pair the model emits a binary mask **P** against ground truth **G**;
IoU = |P∩G| / |P∪G|.

- **gIoU** — mean per-pair IoU. Every pair weighted equally, so a tiny part counts as much as a
  whole object. Convention: IoU = 1 when both are empty.
- **cIoU** — `Σ|Pᵢ∩Gᵢ| / Σ|Pᵢ∪Gᵢ|`, one global area-weighted ratio, dominated by large masks.

Reporting both is diagnostic: a model that only nails big objects looks good on cIoU and bad on
gIoU.

**The decision.** MMR's official code scores **per-target**, with the number of `[SEG]` tokens
teacher-forced to the ground-truth count. That protocol cannot be applied to SAM3, which emits an
*unordered set* of masks with no correspondence to GT targets. So all cross-model numbers here use
a **per-question union** mask — identical to MMR's metric for single-target questions, and the only
formulation both model families can be scored under.

Consequence, stated plainly: **our union numbers are not comparable to the numbers printed in the
paper.** Where we quote the paper we quote its per-target figures; where we compare models we use
our own union recomputation of *both* sides.

## 3. Harness validation

Before trusting any comparison we reproduced M²SA-7B under MMR's official protocol.

| | gIoU | cIoU |
|---|--:|--:|
| Paper, M²SA-7B (per-target, teacher-forced) | 27.8 | 48.6 |
| **Our run of M²SA-7B, same protocol** | **28.1** | **49.0** ✓ |

Within 0.3/0.4 — the harness, the GT decoding and the metric implementation are sound. Both models
are scored by the same `mmr_common.py` (GT decode + meters) from two different conda environments,
so the ground truth is byte-identical across them.

For reference, the paper's published table (per-target, **not** comparable to our union numbers):

| Method | val gIoU | val cIoU | Obj | Part | Obj&Part |
|---|--:|--:|--:|--:|--:|
| LISA-7B | 13.8 | 18.3 | 23.5 | 6.6 | 14.5 |
| LISA-7B<sub>tr</sub> | 19.4 | 31.6 | 34.7 | 8.0 | 19.5 |
| **M²SA-7B** | **27.8** | **48.6** | 41.0 | 13.5 | 30.9 |
| **M²SA-Llama2-13B** | **28.4** | **49.1** | 42.3 | 13.6 | 31.6 |

The paper's best overall is the **13B** at 28.4. We only ever ran the 7B, so nothing here speaks to
the 13B.

### The LISA reproduction does not close

We also ran **LISA-7B** (`xinlai/LISA-7B-v1`), the benchmark's main baseline and the model M²SA is
forked from, through the identical harness. Under MMR's official teacher-forced per-target protocol
we get **8.94** against the paper's **13.8**, and we could not close that gap. Reported as-is.

What we checked, and why we still believe the driving code is right:

- **Config and prompt match LISA's own `chat.py`** — `mm_use_im_start_end: True` in the released
  config, the `llava_v1` conversation template, and the same `<im_start><image><im_end>` wrapping.
- **Free generation agrees with teacher forcing.** Gen mode uses a completely different code path
  (`generate`, no teacher forcing, `[SEG]` positions derived from the model's *own* output) and
  scores 14.44 union vs the teacher path's 12.50. A mis-indexed `[SEG]` position in the teacher
  path would have made the two diverge wildly; they do not.
- **The same harness reproduces M²SA to within 0.3**, so the GT decode, the meters and the metric
  are not at fault.

The most likely explanation is a difference in how the MMR authors adapted a **single-target model
to a multi-target protocol**. LISA emits one `[SEG]` by design; teacher-forcing an N-`[SEG]` answer
is out of distribution for it, and there is more than one defensible way to score that. Since
48.8 % of LISA's teacher-forced predictions come out at *exactly* 0.000 IoU, the metric is extremely
sensitive to that choice.

**Consequence for this document:** LISA's teacher-forced number is treated as a **lower bound** and
flagged wherever it appears; its **free-generation** score is used as the primary LISA data point.

### PixelLM is run free-generation only, on purpose

PixelLM does not use a single `[SEG]` token. The released checkpoint runs
`--seg_token_num=3 --image_feature_scale_num=2`, i.e. a **codebook of 6 tokens** (`[SEG0]..[SEG5]`)
per target, confirmed at load time (`seg_token_idx=[32003..32008]`). MMR's `text_answers` carry
exactly one `{seg}` placeholder per target, so teacher forcing would require inventing a 1→6
expansion — precisely the kind of adaptation choice that produced the unresolved LISA discrepancy
above. Free generation needs no such choice, so it is the only protocol we run for PixelLM.

Prompt parity is exact: PixelLM's own `LONG_QUESTION_LIST` template is
`"{sent} Please output segmentation mask."` — byte-identical to the string given to LISA and M²SA.
No model is prompted outside its native convention. Its *image* pipeline does differ (a 448 CLIP
branch with pad-to-square and `resize_vision_tower`, alongside the usual 1024 mask branch), so the
runner mirrors `chat.py` rather than reusing the LISA path.
That is the honest setting anyway — no answer leak — and it is the one directly comparable to
M²SA-gen and to SAM3 on the raw question.

## 4. Main results

All rows are our own runs, all union-metric, all on MMR val (n = 8,194).

| Model | Input | gIoU | cIoU | Obj | Part | Obj&Part |
|---|---|--:|--:|--:|--:|--:|
| SAM3 base | question | 0.6 | 1.5 | 0.4 | 0.3 | 1.1 |
| SAM3 + SA-1B reasonseg fine-tune | question | 11.1 | 10.6 | 11.8 | 8.4 | 14.9 |
| LISA-7B | question, **teacher-forced** | 12.50 | 19.15 | 16.40 | 9.18 | 15.72 |
| LISA-7B | question, free-generation | 14.44 | 18.85 | 17.04 | 11.54 | 17.64 |
| PixelLM-7B | question, free-generation | 19.21 | 21.77 | 23.86 | 13.66 | 25.47 |
| M²SA-7B | question, free-generation | 25.2 | 30.1 | 31.6 | 19.8 | 30.4 |
| M²SA-7B | question, **teacher-forced** | 36.0 | 54.7 | 39.2 | 30.2 | 43.4 |
| **SAM3 + MMR train (100 %)** | **question** | **37.15** | 36.89 | 45.56 | 27.75 | 47.53 |
| SAM3 base | *oracle concept* | 38.8 | 50.8 | 46.6 | 27.5 | 52.5 |

Reading it:

1. **Base SAM3 cannot do this task at all** (0.6). That is the honest starting point.
2. **Teacher forcing is worth ~10.8 gIoU to M²SA** (36.0 vs 25.2). It is not a deployable setting —
   it feeds the ground-truth answer text, which names the targets — so the fair comparison against
   any question-only model is 25.2.
3. **Teacher forcing *hurts* LISA** (12.50 vs 14.44 free-generation) — the opposite sign to M²SA.
   LISA emits a single `[SEG]` by design; forcing it to produce N masks for an N-target answer it
   was never trained to write yields extra masks that pollute the union. This is the multi-target
   gap MMR was built to expose, and it is visible as a *negative* teacher-forcing bonus.
4. **The zero-shot LMMs order LISA < PixelLM < M²SA** (14.4 → 19.2 → 25.2), reproducing the
   paper's qualitative ordering and extending it. Only M²SA saw MMR in training, so the LISA →
   PixelLM step (+4.8) is a clean architecture comparison at equal (zero) MMR exposure.
5. **PixelLM's multi-target design shows up exactly where it should.** Splitting by target count,
   the single→multi delta is +3.91 for PixelLM but only +0.57 for LISA:

   | model | single-target | multi-target | Δ |
   |---|--:|--:|--:|
   | LISA-7B | 14.08 | 14.65 | +0.57 |
   | **PixelLM-7B** | 16.73 | **20.64** | **+3.91** |
   | M²SA-7B | 24.87 | 25.37 | +0.50 |

   PixelLM is the only model whose advantage concentrates on multi-target questions — which is what
   its segmentation codebook and token-fusion were built for. LISA, single-target by construction,
   gains essentially nothing from extra targets. This is the multi-target axis MMR exists to probe,
   isolated from MMR-specific training.
6. **SAM3 trained on MMR beats all of them**, from the raw question with no answer leak.
7. **The oracle-concept row is the ceiling that explains everything.** Handed the target *name*,
   base SAM3 scores 38.8 — better than M²SA teacher-forced. SAM3 was always the better segmenter;
   it simply could not read the question. Training closes that gap to within **1.65** of the
   ceiling.
8. **cIoU tells a different story than gIoU.** M²SA teacher-forced leads on cIoU (54.7 vs 36.9)
   while trailing on gIoU. cIoU is area-weighted, so this says M²SA is relatively stronger on
   large-mask questions and our model is stronger on the average question. Worth not hiding.

### Where the remaining error is

At 100 %: objects **45.56**, parts **27.75**. Base SAM3 with a perfect prompt was 46.6 / 27.5 — so
training lifts question-level performance up to the oracle *object* level, while parts stay pinned
at almost exactly the value they had with a perfect prompt. **Parts are a limit of the segmenter,
not of the language interface**, and no amount of MMR data moved them. Multi-target questions score
higher than single-target ones (40.48 vs 31.41), which is a union-metric artifact: unioning many
targets produces a larger, more forgiving mask.

## 5. What we trained on

**MMR train, and nothing else.** No semantic-segmentation, referring-expression or VQA data.

| | |
|---|--:|
| Images | 45,618 |
| (image, question) samples | 154,124 |
| Target masks | 272,537 |
| Questions per image | 3.38 mean, 3 median, **5 max** |
| Masks per image | 5.97 mean, 6 median, **42 max** |
| Images missing from disk | 0 |

Converted to SAM3-COCO by `convert_mmr_full.py`: each sample becomes one COCO *category* (`name` =
the question) with one annotation per target mask, so multi-target questions stay multi-instance.
Sample order is a seed-0 shuffle. Images come from COCO train2017; all 45,618 were downloaded.

The SAM3 loader makes one datapoint per **image** and hands the model every query on it, so one
training step covers ~3.38 questions sharing a single vision-backbone pass.

## 6. Training details

Stock **`facebook/sam3`** release checkpoint, fine-tuned end-to-end. Recipe inherited from the
project's `newsam_base` config.

| | |
|---|---|
| Init | `facebook/sam3` release checkpoint |
| Epochs | **1** (one pass over the whole training set) |
| Optimizer steps | **45,618** (one per image) |
| Batch | 1 image / step (~3.38 queries), no gradient accumulation |
| Resolution | 1008 × 1008 |
| Precision | bf16 autocast; GradScaler **off** (bf16 needs no loss scaling) |
| Optimizer | AdamW, weight decay 0.1 (0 on biases / LayerNorm) |
| LR (transformer / vision bb / language bb) | 8e-5 / 2.5e-5 / 5e-6 (base × `lr_scale` 0.1) |
| Vision-backbone layer-wise LR decay | 0.9 |
| Schedule | inverse-sqrt, 20-step warmup, 20-step cooldown, timescale 20 |
| Gradient clipping | max-norm 0.1 |
| Losses | focal (α=0.25, γ=2) + Dice + box L1/GIoU + presence BCE, one-to-many matching (w=2.0), semantic presence/seg/Dice heads |
| Negatives | none (`include_negatives: false`) |
| Hardware | **1 × RTX 3090** |
| Wall clock | **15 h 33 m**, 1.16 s/step avg (incl. 11 in-run benchmark pauses) |

### The ladder: one model, benchmarked in flight

Rather than train N models on N data fractions, we trained **one** model straight through and paused
it to evaluate MMR val at a doubling ladder of data fractions:

> **0.25 → 0.5 → 1 → 2 → 4 → 8 → 16 → 32 → 64 → 100 %**

One optimizer, one LR schedule, one rolling checkpoint overwritten in place — **no per-rung
checkpoint copies**. Each rung is the same weights further along, not a separate run. Because a
1-epoch run is exactly one pass over the data, *fraction of the epoch = fraction of the data*, so
the rung trigger is just a step count.

This is deliberately different from an earlier 4-stage curve in this repo
([`out/mmrcomp/scaling_curve.md`](out/mmrcomp/scaling_curve.md)), which chained *separate* training
jobs and therefore restarted the optimizer and LR warmup at every stage. That design produced a
peak at 2K samples followed by a decline; the restarts were the cause, and removing them changes
the conclusion entirely (§7).

## 7. Data-scaling results

| % of data | images | samples | effective updates | gIoU | cIoU | IoU@50 | Obj | Part | Obj&Part |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 (base) | 0 | 0 | — | 0.6 | 1.5 | 0.6 | 0.4 | 0.3 | 1.1 |
| 0.25 % | 114 | 385 | 114 / 114 | 22.29 | 20.17 | 21.71 | 24.28 | 15.78 | 31.28 |
| 0.5 % | 228 | 770 | 228 / 228 | 26.00 | 23.54 | 24.93 | 28.28 | 18.90 | 35.77 |
| 1 % | 456 | 1,541 | 456 / 456 | 26.09 | 22.60 | 23.97 | 28.14 | 19.33 | 35.45 |
| 2 % | 912 | 3,081 | 912 / 912 | 28.88 | 25.96 | 27.14 | 34.47 | 20.96 | 38.33 |
| 4 % | 1,825 | 6,166 | 1,825 / 1,825 | 30.85 | 27.91 | 29.30 | 36.94 | 22.09 | 41.36 |
| 8 % | 3,649 | 12,328 | 3,649 / 3,649 | 32.16 | 29.67 | 31.28 | 39.19 | 23.47 | 42.10 |
| 16 % | 7,299 | 24,660 | 7,299 / 7,299 | 34.22 | 33.19 | 33.07 | 42.40 | 24.89 | 44.59 |
| 32 % | 14,598 | 49,320 | 14,598 / 14,598 | 35.55 | 35.44 | 34.90 | 44.69 | 26.40 | 45.20 |
| 64 % | 29,196 | 98,641 | 29,196 / 29,196 | 36.88 | 36.53 | 36.25 | 45.53 | 27.72 | 46.77 |
| **100 %** | 45,595 | 154,046 | 45,595 / 45,595 | **37.15** | 36.89 | 36.44 | 45.56 | 27.75 | 47.53 |

A final end-of-epoch evaluation after all 45,618 steps (including the LR cooldown to zero) scores
**36.98** — the same point within noise.

`effective updates` is the count of optimizer steps that were actually *applied* versus attempted.
It reads `n/n` on every row. That column exists because of §9.

![scaling ladder](out/mmrcomp/scaling_curve_full.png)

### Findings

**1. Scaling is monotonic to 100 %. There is no plateau.** Every rung beats the one before, from
22.3 at 385 samples to 37.15 at the full 154K, settling at roughly **+1.3 gIoU per doubling** from
2 % onward:

| doubling | Δ gIoU | | doubling | Δ gIoU |
|---|--:|---|---|--:|
| 0.25 → 0.5 % | +3.71 | | 8 → 16 % | +2.06 |
| 1 → 2 % | +2.79 | | 16 → 32 % | +1.33 |
| 2 → 4 % | +1.97 | | 32 → 64 % | +1.33 |
| 4 → 8 % | +1.31 | | 64 → 100 % (1.6×) | +0.27 |

This overturns the earlier staged curve's plateau-then-decline, which was an artifact of restarting
the optimizer each stage. With one continuous schedule, MMR data keeps paying off to the end of the
training set — the benchmark is not saturated by its own training split.

**2. Extreme efficiency at the low end.** 385 samples (0.25 % of the data) already gives 22.3 —
**59 % of the entire gain over base**. 770 samples (0.5 %) reaches 69 %, and matches M²SA's honest
question-only score. The remaining 99.5 % of the data buys the last 31 %.

**3. There is a reproducible flat spot between 0.5 % and 1 %** — 26.00 → 26.09 here, and
26.70 → 26.59 in the earlier run. Both runs stall in the same place, but the step is well inside
run-to-run variation (§11), so it is worth noting and not worth explaining.

**4. The language gap closes; the part gap does not.** Covered in §4.

### How the learning rate decayed over the data

One step is one image, so a single continuous schedule makes LR a direct function of data seen:
20-step linear warmup to peak, then `lr = peak · √(20 / step)`, then linear cooldown to zero over
the final 20 steps. Since `lr ∝ 1/√step` and `step ∝ data`, **every doubling of the data multiplies
the LR by 1/√2 = 0.707** — a constant factor per rung, by construction.

| rung | transformer LR | % of peak | cumulative LR-mass |
|--:|--:|--:|--:|
| 0.25 % | 3.35e-05 | 41.9 % | 3.5 % |
| 1 % | 1.68e-05 | 20.9 % | 8.6 % |
| 4 % | 8.38e-06 | 10.5 % | 18.7 % |
| 16 % | 4.19e-06 | 5.2 % | 39.1 % |
| 32 % | 2.96e-06 | 3.7 % | 55.9 % |
| 64 % | 2.09e-06 | 2.6 % | 79.7 % |
| 100 % | 0 (cooldown) | 0 % | 100 % |

Peak LR is reached after only **20 images (0.04 % of the data)**, so the entire ladder lives on the
decaying tail — even the first rung is already at 42 % of peak. "Cumulative LR-mass" (the sum of LR
over steps, a proxy for total parameter movement) grows as **√data**: by 32 % of the data the model
has already spent **56 %** of its total parameter movement, and the last 36 % of the data
contributes only 20 %.

**This is the main caveat on the ladder.** Later rungs train at much lower LR, so the tail-end
flattening (+0.27 over the final 1.6×) is partly schedule, not purely data saturation. The
monotonicity is not in doubt, but the ladder is properly read as a **continual-training curve**
rather than a data-efficiency curve. Separating the two needs a rung-sized schedule (each fraction
with its own warmup and decay) or a constant LR — the obvious follow-up.

## 8. Did we out-spend the paper?

**No — they used ~2.4× more MMR supervision than we did.** M²SA's `--epochs 10` is a misnomer:
`steps_per_epoch` is a fixed step count, not a pass over the data, and MMR is one of four datasets
in a LISA-style mixture (`sem_seg||refer_seg||vqa||multi_part_reason_seg` at sample rates `2,9,2,6`,
so MMR draws are 6/19 = **31.6 % of the training mixture** — that is the share of the mixture, not
the share of MMR used).

```python
samples_per_epoch = batch_size(2) × grad_accumulation_steps(10)
                  × steps_per_epoch(500) × world_size(4)      # = 40,000
```

× 10 epochs = 400,000 draws, of which 31.6 % → **126,316 MMR draws**. Two details of
`MultiPartReasonSegDataset.__getitem__` decide what that is worth:

- it **ignores the index and samples an image uniformly at random with replacement**
  (`idx = random.randint(0, len(...)-1)`), so draws are not a clean pass over the split;
- one draw is **one image carrying up to `num_classes_per_sample = 3` questions**, not one
  question. With MMR's question histogram (`{1:57, 2:397, 3:35746, 4:1055, 5:8363}`),
  E[min(3, q)] = 2.989 questions per draw.

| | M²SA-7B (paper) | SAM3 (ours) |
|---|--:|--:|
| MMR image-draws | 126,316 (**2.77 passes**) | 45,618 (1.00 pass) |
| MMR question-instances seen | **~377,500** (with repeats) | 154,124 (each exactly once) |
| Unique MMR images touched | ~42,757 (93.7 %) | 45,618 (**100 %**) |
| Total draws, all datasets | 400,000 | 154,124 |
| Optimizer steps | 5,000 | 45,618 |
| Global batch | 80 | 1 image (~3.4 questions) |
| Peak LR | 3e-4 | 8e-5 |
| Trainable | LoRA r=8 + mask decoder, on LLaVA-7B | full model |
| GPUs | 4 | 1 |

So on every axis of *data* they had the advantage: ~2.4× the MMR question-instances, plus ~274K
samples of semantic-seg / referring-seg / VQA data we never touched, on top of LLaVA-7B's
vision-language pretraining. Our single advantage is coverage — one clean pass touches 100 % of the
split, where random sampling with replacement leaves ~6 % of images unseen.

The one caveat in their favour: **they trained LoRA adapters (r=8) plus the mask decoder, not the
full model**, which is a real handicap on a new task. That is a choice, not an imposed constraint,
but it means the comparison is "full fine-tune of a strong segmenter" vs "adapters on a strong
language model", not a like-for-like capacity match.

The summary: with **less** MMR supervision and no auxiliary datasets, a stronger segmentation
backbone beats a weaker backbone driven by a 7B language model.

## 9. Inference cost

Measured on one RTX 3090, bf16, over real MMR val pairs. Model load, CUDA context and autotune are
excluded via warmup; `torch.cuda.synchronize()` brackets every timed region; images are decoded into
RAM up front so no disk I/O sits inside the measurement.

The two model families divide work differently, and folding that in wrongly is the easy way to get
an unfair ratio. SAM3 encodes the image **once per image** (111 ms) and runs text encoder + decoder
**per prompt** (~66 ms), reusing the image encoding across that image's questions. M²SA recomputes
CLIP + SAM ViT-H inside **every** forward — nothing to amortize. SAM3 is therefore reported both
cold and amortized at MMR val's real 3.408 questions/image.

| Model / setting | image enc | per-prompt fwd | cold | **amortized** | throughput |
|---|--:|--:|--:|--:|--:|
| M²SA-7B — question, free-generation | — | 4220.1 | 4220.1 | **4220.1** | 0.24 /s |
| M²SA-7B — teacher-forced | — | 340.0 | 340.0 | **340.0** | 2.94 /s |
| LISA-7B — question, free-generation | — | 1185.6 | 1185.6 | **1185.6** | 0.84 /s |
| LISA-7B — teacher-forced | — | 345.1 | 345.1 | **345.1** | 2.90 /s |
| SAM3 — question | 111.3 | 67.0 | 178.3 | **99.7** | 10.03 /s |
| SAM3 — oracle concept | 111.8 | 100.6 | 212.3 | **133.4** | 7.50 /s |

SAM3 latency was measured on the base and reasonseg checkpoints, not the MMR-trained one; the
architecture is identical across them and base vs fine-tuned differ by ~1 ms, so the figures carry
over.

LISA and M²SA are the same architecture family, and their teacher-forced (single prefill) costs are
within 2 % of each other — 345 vs 340 ms — as they should be. The gap opens only in generation:
M²SA writes longer multi-target answers (~101 tokens) than LISA (~79) and is **3.6× slower**
(4220 vs 1186 ms), so most of M²SA's extra cost is autoregressive decoding, not vision.

**M²SA free-generation is 42× slower per prediction than SAM3** (LISA-gen is 12× slower). It is also the only variable one
(median 4220, mean 4426, p90 6437, min 2328 ms) because it is autoregressive over ~101 tokens;
SAM3's p90 is within 4 % of its median. Mask→CPU transfer is 0.01–0.4 ms for both and M²SA's CPU
preprocessing is 28 ms/image — neither is folded into the figures above.

Combining with §4: our model is **+11.9 gIoU over M²SA question-only at 1/42 the latency**, and
fine-tuning is latency-neutral (99.7 vs 98.4 ms for base).

## 10. What went wrong

Reported because it changes how much to trust the numbers, and because the failure mode is easy to
repeat.

### Run 1 silently stopped learning

The first full ladder run completed all 45,618 steps and produced a plausible-looking curve that
plateaued at 32 %. It was wrong. **25,248 of 45,618 optimizer steps (55 %) had been dropped** by the
trainer's non-finite-gradient guard:

| segment | steps | dropped | |
|---|--:|--:|---|
| 0 → 16 % | 7,299 | 0 | clean |
| 16 → 32 % | 7,299 | 4 (0.1 %) | clean |
| 32 → 64 % | 14,598 | 8,823 (60 %) | contaminated |
| 64 → 100 % | 16,399 | **16,399 (100 %)** | frozen |

From ~step 25,000 every step was skipped, so the weights stopped moving entirely. The proof was
direct: the checkpoints evaluated at the 64 % and 100 % rungs were **byte-identical**, which is why
those two rows reported the same numbers to the last decimal. The apparent "plateau" was a frozen
model. Meanwhile the loss log looked entirely healthy (~160, finite, right to the end) — that is
what makes this failure mode dangerous.

### Root cause: a fused Triton focal-loss kernel

`torch.autograd.set_detect_anomaly` named it:

```
RuntimeError: Function 'SigmoidFocalLossReducedBackward' returned nan values in its 0th output.
  ... loss_fns.py:432 get_loss    <- IABCEMdetr presence loss
```

The kernel's backward forms `alpha_t · (d_bce·mod + d_mod·bce_loss)` where
`d_mod = -γ·d_pt·tmp` and `d_pt = (2t-1)·σ·(1-σ)`. As a head saturates, `d_pt` underflows to
exactly 0 while `bce_loss` grows with |logit|, so the term becomes `0 × huge`; separately, the
kernel's recomputed BCE clamps `max_val` at `1e9`, breaking the identity for large logits. The
failing call was the **presence loss**, whose head had reached `presence_dec_acc = 0.9998` — exactly
the saturated regime. The stock PyTorch path in the same function builds on
`binary_cross_entropy_with_logits` and has neither problem.

A/B from the frozen end-of-run weights, which reproduce the failure on demand:

| variant | skipped steps |
|---|--:|
| fused Triton focal (original) | **130 / 309** |
| GradScaler disabled under bf16 | 136 / 315 — *no effect* |
| **stable PyTorch focal path** | **0 / 305** |

Step time is unchanged at 1.10 s/step, so the fused kernel was buying nothing here.

**Two things this was not.** Nothing diverged — comparing final weights against base tensor by
tensor, the largest relative change is 0.66 on one small bias and everything else is ≤ 0.15, with no
NaN or Inf anywhere. And `text_projection` at `|max| = 9.6e18` is a red herring: the **stock
`facebook/sam3` checkpoint ships exactly that value**, unchanged by training. The GradScaler
(constructed with `enabled=amp.enabled` while `amp_dtype` is bf16) is a genuine footgun and is now
disabled under bf16 on principle, but the A/B shows it was not the culprit.

**Why nothing hit this before:** the first skipped step is iter 8,180, and every previous run on
this recipe was shorter — the SA-1B reasonseg fine-tune was 8,144 steps, the earlier MMR staged runs
≤ 3,867. This was simply the first run long enough for the presence head to saturate. The
instability had been latent in the recipe all along.

### Two performance findings that shaped the setup

**Three GPUs were 15× slower per sample than one.** DDP across all 3 RTX 3090s cost ~16 s/step of
gradient all-reduce for ~843M fp32 parameters: GeForce 30-series cards have no P2P over PCIe and the
topology here is PHB. Training runs on one GPU; the other two stay free to shard each rung's
benchmark.

| layout | GPUs | s/step | samples/step | **s per sample** |
|---|--:|--:|--:|--:|
| grouped | 3 (DDP) | 17.00 | 3.38 | 5.03 |
| ungrouped | 3 (DDP) | 17.00 | 1.00 | 17.00 |
| ungrouped | 1 | 1.04 | 1.00 | 1.04 |
| **grouped** | **1** | **1.10** | **3.38** | **0.33** |

**Grouping questions by image is 3.2× cheaper per sample.** The 1008² vision backbone dominates and
is shared across an image's 3.38 questions. Giving each (image, question) pair its own datapoint
costs 1.04 s/sample; grouping costs 0.33. The full run is ~14 h instead of ~44 h.

A third, smaller fix: the COCO loader built its per-datapoint query list by scanning the *entire*
category vocabulary — fine at 4K categories, ~1 s/step at 154K. With `include_negatives=false` every
empty category is skipped anyway, so it now walks only the image's own categories.

### Guards added, and verified to fire

| layer | catches |
|---|---|
| trainer self-abort | > 50 % of the last 200 steps skipped → kills the run |
| grad diagnostics | first few NaN events log which parameter tensors, grouped by module |
| per-rung weight fingerprint | logs `WEIGHTS UNCHANGED` if a rung is bit-identical to the last |
| periodic skip log | rolling skip count + grad-scale every 500 steps |
| `guard_ladder.sh` | external watchdog: alerts on skips, unchanged weights, stalled log |
| `plot_ladder.py` | strikes through rungs with too few effective updates, excludes them from the plot |

Writing the guards surfaced two bugs in the guards themselves, both found by testing rather than
inspection:

- The abort raised `FloatingPointError`, which the per-batch handler in `train_epoch` catches to
  ride out one-off bad batches — it would have been swallowed. It now raises `RuntimeError`.
- The weight fingerprint summed all parameter norms into one float. With `text_projection` at
  ~1e19 the sum sits where one ULP is ~2048, while the entire rest of the model contributes ~5e3 —
  it would have reported "unchanged" for essentially any real update. It now compares per-tensor.

`out/mmrcomp/code/test_guards.py` exercises the real `Trainer` methods and asserts each guard fires
(and that a 40 % skip rate does *not* trip the abort). Run 2 finished with **zero skipped steps and
zero alerts**, and every ladder rung logged 1,022–1,068 of 1,102 parameter tensors moving (the extra
end-of-epoch check, only 23 steps after the previous rung and during the LR cooldown, logged 609).

## 11. Limitations

- **Single run per configuration**, so no formal seed variance. The two full runs we have do give
  a usable estimate over the eight rungs where both were valid: **mean |Δ| = 0.30 gIoU, max 0.70**
  (largest at the 0.5 % rung, 26.70 vs 26.00). Any inter-rung difference below ~0.7 should
  therefore not be read as signal — which covers the 0.5 %/1 % flat spot, though not the ~1.3
  per-doubling trend. Caveat on that estimate: the two runs also differ in focal-loss
  implementation, so it bundles numerical differences with seed noise.
- **The LR schedule is confounded with the ladder** (§7). This is inherent to "one continuously
  trained model" and is the honest continual-learning reading, but it is not a clean
  data-efficiency measurement.
- **Union metric only** for cross-model comparison. Not comparable to the paper's per-target
  numbers, and it flatters multi-target questions.
- **Val split only.** MMR also has testA/testB splits (32,077 pairs) which we never touched, and
  no result here is a held-out test number.
- **7B only.** The paper's best model is the 13B (28.4 per-target); we never ran it.
- **Three baselines, not a field survey.** We benchmarked LISA-7B, PixelLM-7B and M²SA-7B. Other
  reasoning-segmentation systems (GSVA, GLaMM, SESAME, u-LLaVA, OMG-LLaVA) are *not* covered. Each needs its own repo, environment and runner, plus a
  16–43 GB download, and roughly an hour of eval per protocol; LISA was cheap only because M²SA is
  a fork of it, so the existing runner transferred with a two-line change; PixelLM needed a
  purpose-built runner. Weights are public for GLaMM (16.8 GB), SESAME (16.2 GB) and LISA-13B
  (28.7 GB) if this is worth extending.
- **PixelLM and LISA are zero-shot on MMR**, while M²SA and our SAM3 saw MMR train. Comparisons
  across that line measure training exposure as much as architecture; the LISA↔PixelLM comparison
  does not, which is why it carries the multi-target conclusion.
- **LISA's teacher-forced number does not reproduce** the paper's (8.94 vs 13.8) and is a lower
  bound; see §3.
- **In-distribution training.** SAM3 is trained on MMR train and evaluated on MMR val. Whether
  this transfers to other reasoning-segmentation benchmarks is untested, and the SA-Co/Gold result
  in the main [`README.md`](README.md) is a caution: a narrow fine-tune regressed out-of-domain
  through over-triggering. We did not re-measure SA-Co/Gold for this model.
- **No ablations** on which part of MMR does the work (objects vs parts, single vs multi-target),
  nor on freezing the language backbone.
- **The 37.15 weights no longer exist.** The ladder overwrote one rolling checkpoint in place, so
  what is on disk is the end-of-epoch state at 36.98. [`quant.md`](quant.md) §4 reproduces that
  number exactly with a rebuilt loader (`/workspace/newsam`, which `run_sam3_mmr.py` imported
  `build_model` from, is gone), which also re-validates this document's harness.

## 12. Reproducing

```bash
# 1. MMR train -> one SAM3-COCO json (45,618 images / 154,124 samples)
python out/mmrcomp/code/convert_mmr_full.py

# 2. the ladder: one continuous run that benchmarks itself at every rung (~15 h, 1 GPU)
bash out/mmrcomp/code/run_scale_ladder.sh

# 3. watchdog + live table/plot refresh
bash out/mmrcomp/code/guard_ladder.sh &

# 4. verify the guards actually fire
python out/mmrcomp/code/test_guards.py
```

| file | what it does |
|---|---|
| `out/mmrcomp/code/mmr_common.py` | MMR val loading, GT decode, union + native meters, granularity split |
| `out/mmrcomp/code/convert_mmr_full.py` | MMR train → SAM3-COCO |
| `sam3/train/configs/newsam/mmr_scale.yaml` | the run: 1 epoch, batch 1, rolling checkpoint only |
| `out/mmrcomp/code/run_scale_ladder.sh` | sets the rung fractions and launches |
| `out/mmrcomp/code/bench_scale.sh` | one rung: 3-way sharded MMR-val eval |
| `out/mmrcomp/code/run_sam3_mmr.py` | SAM3 on MMR val (`--prompt-mode question|concept`) |
| `out/mmrcomp/code/run_m2sa_mmr.py` | M²SA-7B (`--mode teacher|gen`), m2sa env, cwd = MMR repo |
| `out/mmrcomp/code/run_lisa_mmr.py` | LISA-7B, same interface, m2sa env, cwd = LISA repo |
| `out/mmrcomp/code/run_lisa_wave.sh` | both LISA protocols, 3-way sharded, merged |
| `out/mmrcomp/code/run_pixellm_mmr.py` | PixelLM-7B, free generation; 448 CLIP + 1024 mask pipeline |
| `out/mmrcomp/code/run_pixellm_wave.sh` | PixelLM 3-way sharded, with flush/resume/retry |
| `out/mmrcomp/code/bench_{sam3,m2sa,lisa}_latency.py` | per-prediction latency |
| `out/mmrcomp/code/plot_ladder.py` / `plot_lr.py` | ladder table + curve, LR overlay |
| `out/mmrcomp/code/guard_ladder.sh` / `test_guards.py` | watchdog and its tests |

Per-rung metric dumps are in `out/mmr_eval/mmrscale_*.json`; run 1's invalid dumps are kept in
`out/mmr_eval/run1_nan/` with its curve as `out/mmrcomp/run1_scaling_curve.png`. Longer-form
versions of individual sections live in [`out/mmrcomp/README.md`](out/mmrcomp/README.md) (head-to-
head), [`scaling_ladder.md`](out/mmrcomp/scaling_ladder.md) (ladder) and
[`latency.md`](out/mmrcomp/latency.md) (cost).
