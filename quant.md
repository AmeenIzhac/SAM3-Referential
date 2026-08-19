# Quantizing the MMR SAM3 checkpoint

Weight-only post-training quantization of the SAM3 checkpoint from [`mmr.md`](mmr.md) — the one
trained on MMR train that reaches **36.98 gIoU** on MMR val from the raw question — re-evaluated at
every width from 8 bits down to 2 on the full val split (2,404 images / 8,194 questions).

**Headline.** **8, 6 and 5 bits are free**: 36.97 / 36.96 / 36.94 gIoU against an fp32 baseline of
36.98, i.e. inside ±0.04 on a benchmark whose own run-to-run spread is ~0.3–0.7 (mmr.md §11). **4
bits costs 0.41 gIoU** (36.57, 98.9 % of baseline) for a **6.7× smaller checkpoint**, and recovers to
**−0.13** by grouping the scales more finely. The cliff is at **3 bits** (33.02, −3.96 — still ahead
of every question-only baseline in mmr.md) and the collapse at **2 bits** (0.35, *below* base SAM3's
0.6: the fine-tune is gone entirely).

Three things found on the way matter more than the curve:

- a module-type scan silently misses **83M parameters** of attention weight (§2);
- the stock `facebook/sam3` checkpoint carries a tensor that turns an fp16 quantization scale into a
  **silent NaN** (§3);
- the **detection heads are 3.3 % of the parameters but 21× more precision-sensitive per parameter**
  than the vision backbone (§6) — and a finer scale group is nonetheless a better buy than protecting
  them, which is the opposite of what that sensitivity suggests (§6).

This is a **checkpoint-size and robustness** result, not a speed one. Nothing here makes SAM3
faster — see §9.

---

## 1. What "quantized" means here

Post-training **round-to-nearest (RTN)**, weight-only, group-wise, asymmetric. No calibration data,
no GPTQ/AWQ error compensation, no activation quantization: the honest floor that any smarter method
has to beat.

Each 2-D weight `(out, in)` is split along the **input** dim into groups of `group_size`. Each group
gets its own scale and integer zero-point, fitted to that group's min/max with 0 forced into the
range:

```
scale = (max(w,0) - min(w,0)) / (2**bits - 1)          # per group
zero  = round(-min(w,0) / scale)                        # uint8
q     = clamp(round(w/scale) + zero, 0, 2**bits - 1)    # the stored code
ŵ     = (q - zero) * scale
```

Forcing 0 into every group's range means **w = 0 is exactly representable**, so zero-valued weights
never drift (tested at every width). The scale is rounded to its stored dtype *before* the codes are
fitted, so a save/load round-trip returns the exact tensors that were quantized — `test_wquant.py`
asserts tensor-by-tensor equality across all 1,134 tensors between an in-memory quantization and a
write-then-read of the artifact.

**One caveat on "bit-exact", measured rather than assumed.** That equality holds for a fit performed
on the *same device*. The fit is device-dependent in its last bit: `(max-min)/qmax` rounded to bf16
can land either side of a rounding boundary on CPU versus CUDA, and `round(w/scale)` then shifts.
Measured over three large tensors, **7 codes and 4 scales out of 58.6M differ** (~1e-7) between a CPU
and a GPU fit. `quantize_ckpt.py` fits on the GPU and the on-the-fly eval path fits on the CPU, so
the two differ by exactly that much — which shows up in §5/§6 as **int4 g128 at 36.57 on the fly
versus 36.58 from the artifact**. The artifact rows are the measured score of the file that ships;
if you need the two paths to agree exactly, fit both on the same device.

### Bit-packing at widths no kernel supports

Codes are packed at exactly `bits` bits/weight by a **bit-plane transpose** over groups of 8 values:
8 values × k bits = k bytes, byte *j* holding bit *j* of each of those 8 values. This works for any
k in 1..8, which is why **5- and 6-bit are real here** rather than 5 bits of information stored in
an 8-bit container. Measured, exhaustively over every representable code: 3.00 / 4.00 / 5.00 / 6.00 /
7.00 / 8.00 bits per value at k = 3..8.

Storage overhead is one bf16 scale + one uint8 zero per group, so the true cost is

```
bits/weight = bits + 24 / group_size
```

— 4.19 for int4/g128, 4.75 for int4/g32, 5.19 for int5/g128. The results tables carry this column,
because without it the group-size rows are not size-comparable to the bit-width rows.

### Two runtime modes

| mode | what is resident | speed |
|---|---|---|
| `dense` | ordinary fp32 weights, unpacked once at load | full |
| `packed` | uint8 codes, dequantized inside every forward | 2.13× slower, 3.05× less weight VRAM (§9) |

Both produce **bit-identical weights** (asserted in `test_wquant.py`); `dense` is what every number
in this document was measured with. `packed` exists to show the artifact is self-contained and to
measure resident VRAM.

## 2. Coverage: 98.2 %, and the 83M parameters a module scan misses

Selection runs over **2-D floating-point parameters**, not module types. That choice is not
cosmetic. SAM3's CLIP text encoder is built from `nn.MultiheadAttention`, whose `in_proj_weight` is
a **raw `nn.Parameter`, not an `nn.Linear`** — so the obvious `isinstance(mod, nn.Linear)` scan
skips it, silently, and reports a plausible-looking 82.3 % coverage. There are **61 such modules
holding 82.8M parameters**: 24 in the language backbone (3072×1024 each, **75.5M** — the part of the
model that actually reads the reasoning question), plus 37 smaller ones in the transformer heads,
geometry encoder and segmentation head (768×256, 7.3M).

| selection | tensors | params | coverage |
|---|--:|--:|--:|
| `nn.Linear` weights only | 286 | 692.1M | 82.3 % |
| + 61 `MultiheadAttention.in_proj_weight` | 347 | 774.9M | 92.2 % |
| + the 49408×1024 CLIP token embedding | 348 | 825.5M | **98.2 %** |

The remaining 1.8 % (15.0M params) is left in fp32 on purpose: 8.7M of conv kernels (including the
14×14 patch embedding, a first layer), and ~6.3M of LayerNorms, biases, positional tables and
complex-valued RoPE frequencies. Every one is either under 65,536 elements — where the per-group
scale+zero overhead is a large fraction of the tensor and buys no compression — or not 2-D. They are
also, at 1.8 % of the model, not where the bytes are.

## 3. The tensor that turns an fp16 scale into a NaN

The stock `facebook/sam3` release checkpoint ships
`backbone.language_backbone.encoder.text_projection` — shape (1024, 512), median |w| = 0.019, 99th
percentile 0.085 — containing **exactly two entries at −9.58e18**. Its outlier ratio (max / p99.9)
is **8.3e19×**; the next worst tensor in the model is 18.3×, and the median is 2.64×.

Group-wise quantization handles that badly, and with an fp16 scale it fails *silently*:

```
scale = 9.58e18 / 255 = 3.76e16          # fp32: fine
scale.to(float16)     = inf              # fp16 max is 65504
zero  = round(9.58e18 / inf) = 0
q     = round(0.0425 / inf) + 0 = 0
ŵ     = (0 - 0) * inf = NaN              # <- silent
```

128 weights become NaN. Nothing raises, no loss changes, and the only visible symptom is that
`rel_err_mean` in the run log prints `nan` — which is easy to scroll past.

**Why it did no damage:** SAM3 never reads that parameter. It is
`nn.Parameter(torch.empty(width, output_dim))` (hence the denormals, exact zeros and two identical
garbage values — uninitialized memory that got serialized), it is applied only to the *pooled* text
feature, and `VETextEncoder.forward` discards that: `_, text_memory = self.encoder(tokenized)`
(`sam3/model/text_encoder_ve.py:308`). So the MMR numbers were never affected. mmr.md §10 already
flagged this tensor as "a red herring" for the NaN-gradient investigation; for quantization it is
the exact opposite — the single most hostile tensor in the model, by nineteen orders of magnitude.

**The fix — bf16 scales.** Same 2 bytes as fp16, but bf16 carries fp32's exponent range, so 3.76e16
is an ordinary number and the pathological group dequantizes to finite values (recovering the
outlier and zeroing its 126 neighbours, which is what RTN *should* do with an input like that).
Measured cost of bf16 over fp16 on five real SAM3 tensors: **+0.5 % relative error at 8 bits, +0.0 %
at 4**. On top of that, `quantize_model_inplace` and the serializer now **refuse to return a tensor
containing a non-finite value**, and an unrepresentable scale raises instead of going quiet. Both
guards are exercised by `test_wquant.py`.

## 4. Harness validation

`/workspace/newsam`, which the original `mmrcomp/code/run_sam3_mmr.py` imported `build_model`
from, no longer exists on this box, so the loader was rebuilt against the stock `/workspace/sam3`
tree (the same one `gold_eval.py` uses). The rebuilt harness reproduces the recorded run **exactly**:

| | stored `mmr_eval/mmrscale_epoch_1.json` | rebuilt fp32 |
|---|--:|--:|
| gIoU | 36.98 | **36.98** |
| cIoU | 36.56 | **36.56** |
| IoU@50 | 36.01 | **36.01** |
| Obj / Part / Obj&Part | 45.47 / 27.53 / 47.39 | **45.47 / 27.53 / 47.39** |
| single / multi target | 31.19 / 40.33 | **31.19 / 40.33** |
| n (questions / obj / part / both) | 8194 / 1291 / 4171 / 2732 | **8194 / 1291 / 4171 / 2732** |

Every metric and every count, to the last decimal, so the deltas below are attributable to
quantization and nothing else.

**One note on the baseline.** It is **36.98, not the 37.15** headline in mmr.md. The ladder trained
one model continuously through a **single rolling checkpoint overwritten in place**, so the weights
that scored 37.15 at the 99.95 % rung no longer exist — what is on disk is the end-of-epoch state,
23 steps and an LR cooldown later, which mmr.md §7 already reports at 36.98 and calls "the same
point within noise". Quantization deltas are measured against the checkpoint that exists.

## 5. Accuracy vs bit width

Full MMR val, 8,194 (image, question) pairs, raw reasoning question as the prompt, union metric —
the identical protocol to mmr.md §4. `bits/weight` is the true storage cost including the per-group
scale and zero (§1). All rows are 98.2 % coverage, g128, embeddings included.

<!--BEGIN:RESULTS_MAIN-->
| weights | bits/weight | gIoU | Δ gIoU | cIoU | IoU@50 | Obj | Part | Obj&Part | n |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| fp32 (baseline) | 32 | 36.98 | +0.00 | 36.56 | 36.01 | 45.47 | 27.53 | 47.39 | 8194 |
| bf16 cast | 16 | 36.99 | +0.01 | 36.52 | 36.06 | 45.46 | 27.55 | 47.41 | 8194 |
| **int8** g128 | 8.19 | 36.97 | -0.01 | 36.42 | 36.00 | 45.40 | 27.59 | 47.31 | 8194 |
| int6 g128 | 6.19 | 36.96 | -0.02 | 36.52 | 35.97 | 45.15 | 27.54 | 47.48 | 8194 |
| **int5** g128 | 5.19 | 36.94 | -0.04 | 36.50 | 36.09 | 45.15 | 27.61 | 47.30 | 8194 |
| **int4** g128 | 4.19 | 36.57 | -0.41 | 36.56 | 36.20 | 44.72 | 27.15 | 47.11 | 8194 |
| int3 g128 | 3.19 | 33.02 | -3.96 | 33.90 | 32.63 | 40.21 | 23.75 | 43.78 | 8194 |
| int2 g128 | 2.19 | 0.35 | -36.63 | 1.83 | 0.32 | 0.17 | 0.26 | 0.56 | 8194 |
<!--END:RESULTS_MAIN-->

**Reading it:**

1. **8, 6 and 5 bits are free.** −0.01, −0.02, −0.04 gIoU. mmr.md §11 puts run-to-run spread on this
   benchmark at mean |Δ| 0.30 / max 0.70 gIoU, so all three are far inside the noise the project
   already declared unreadable. **int5 is the best deal**: 5.19 bits/weight, a 5.57× smaller
   checkpoint, and 36.94 vs 36.98.
2. **4 bits costs 0.41 gIoU** — 98.9 % of baseline for 6.71× compression (502 MB). This is the first width
   where the loss is larger than the benchmark's own noise, but only just, and §6 recovers most of it.
3. **3 bits is the cliff: −3.96.** Not a collapse — 33.02 still beats every question-only baseline in
   mmr.md (M²SA-7B 25.2, PixelLM-7B 19.2, LISA-7B 14.4) and is 55× base SAM3's 0.6. A 3-bit
   MMR-trained SAM3 is still the best question-only model in that table.
4. **2 bits is a collapse: 0.35**, *below* base SAM3's 0.6. Everything the fine-tune learned is gone,
   and then some.
5. **The damage is spread across granularities, not concentrated in one.** At int4 the three
   buckets lose a similar small fraction of their own baseline — Obj −0.75 (1.6 %), Part −0.38
   (1.4 %), Obj&Part −0.28 (0.6 %). Quantization does not selectively
   destroy the part-segmentation that mmr.md §4 identified as the model's weak axis; parts were
   limited by the segmenter, and that limit is not a precision limit.
6. **cIoU barely moves until 3 bits** (36.56 → 36.56 at int4, actually unchanged) **while gIoU
   falls.** That divergence is the whole story of §7.

## 6. Where the bits are best spent

Three levers, all measured at the widths where they matter: how finely the scales are grouped,
whether the CLIP token embedding is included, and **which subsystem** is held at higher precision.
Each gets its own table below; per-tensor fit statistics for every config are collected at the end of
the section.

### Group size: at 4 bits, scales are a better buy than codes

| int4 variant | bits/weight | gIoU | Δ |
|---|--:|--:|--:|
| per-channel (one scale per row) | 4.02 | 35.57 | −1.41 |
| g128 | 4.19 | 36.57 | −0.41 |
| g64 | 4.38 | 36.68 | −0.30 |
| g32 | 4.75 | **36.85** | **−0.13** |

At 8 bits granularity is irrelevant — plain **per-channel int8 is free** (37.00, +0.02 at 8.02
bits/weight), so the stock deployment default needs no group-wise machinery at all. At 4 bits it is
decisive, and the marginal return is wildly uneven: the **first 0.17 bits/weight** (per-channel →
g128) buys **+1.00 gIoU**, the next 0.19 buys +0.11, and the next 0.37 buys +0.17. The same pattern
holds at 3 bits (g128 33.02 → g64 34.74 → g32 35.82, +2.80 for 0.56 bits/weight).

The practical reading: **do not run 4-bit per-channel.** It is only 0.17 bits/weight cheaper than
g128 and costs 3.4× the accuracy.

### The CLIP token embedding is safe to quantize

Excluding the 49408×1024 token table (92.2 % coverage instead of 98.2 %) scores 36.63 against
36.57 with it included — **+0.06 gIoU for 6 points of coverage**, about 25 MB at int4. Not worth
protecting; it stays quantized in the shipped artifacts.

### Which subsystem is precision-critical: the heads, by 21×

Quantizing exactly one subsystem to 3 bits and leaving the rest in fp32. The three probes **exactly
partition** the 825.5M quantizable parameters — 444.6M + 353.4M + 27.5M, no overlap and no gap — so
the costs are directly comparable.

| quantized to int3 | params | share | gIoU | Δ | **Δ per 100M params** |
|---|--:|--:|--:|--:|--:|
| vision backbone only | 444.6M | 53.9 % | 35.74 | −1.24 | 0.28 |
| language backbone only | 353.4M | 42.8 % | 35.77 | −1.21 | 0.34 |
| **detection heads only** | **27.5M** | **3.3 %** | 35.32 | **−1.66** | **6.03** |
| all three (= the int3 g128 row) | 825.5M | 100 % | 33.02 | −3.96 | 0.48 |

Three things fall out:

1. **The heads are the bottleneck.** 3.3 % of the parameters cost *more in absolute terms* than
   either backbone, and **21× more per parameter** than the vision backbone. This is independently
   visible in the fit statistics: the worst per-tensor relative error at 8, 6, 5 and 4 bits is
   always a decoder head tensor (`transformer.decoder.ref_point_head.layers.1`), never a backbone
   one. The heads are small, low-redundancy, and produce coordinates and presence logits rather
   than distributed features — exactly the shape of thing RTN handles worst.
2. **Vision and language are equally tolerant, and language only marginally more sensitive per
   parameter** (0.34 vs 0.28, a factor of 1.2). mmr.md's thesis is that SAM3's MMR gain was a
   *language* effect — "the gap between SAM3 and a reasoning-segmentation model was language, not
   pixels". That is a claim about where the *training signal* went, and it does **not** transfer to
   precision: the language backbone is not meaningfully more fragile than the vision backbone. Two
   different senses of "where the model's competence lives", and they do not coincide.
3. **The damage is additive, not compounding.** The three individual costs sum to 4.11 against a
   joint cost of 3.96 — very slightly *sub*-additive. Quantization error in one subsystem does not
   amplify through the vision-language fusion; it just accumulates.

### Mixed precision: it works, and it is still the second-best lever

If the heads carry 21× the sensitivity at 3.3 % of the size, holding *only* them at 8 bits should be
nearly free in storage. It is — int3 body + int8 heads costs 3.35 bits/weight against uniform int3's
3.19 — and it buys a lot: **+1.37 gIoU at int3** (33.02 → 34.39) for 0.16 bits/weight.

<!--BEGIN:RESULTS_MIXED-->
| weights | bits/weight | gIoU | Δ gIoU | cIoU | IoU@50 | Obj | Part | Obj&Part | n |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| int3 g128 | 3.19 | 33.02 | -3.96 | 33.90 | 32.63 | 40.21 | 23.75 | 43.78 | 8194 |
| int3 g128 + **int8 heads** | 3.35 | 34.39 | -2.59 | 33.59 | 33.01 | 41.77 | 24.88 | 45.42 | 8194 |
| int3 g64 | 3.38 | 34.74 | -2.24 | 35.62 | 34.31 | 42.02 | 25.05 | 46.10 | 8194 |
| int3 g64 + **int8 heads** | 3.54 | 35.18 | -1.80 | 34.49 | 33.89 | 42.92 | 25.49 | 46.31 | 8194 |
| int3 g32 | 3.75 | 35.82 | -1.16 | 36.73 | 35.36 | 44.18 | 26.43 | 46.20 | 8194 |
| **int4** g128 | 4.19 | 36.57 | -0.41 | 36.56 | 36.20 | 44.72 | 27.15 | 47.11 | 8194 |
| int4 g128 + **int8 heads** | 4.32 | 36.62 | -0.36 | 36.16 | 35.76 | 44.04 | 27.12 | 47.62 | 8194 |
| int4 g64 | 4.38 | 36.68 | -0.30 | 36.69 | 35.93 | 44.74 | 27.31 | 47.18 | 8194 |
| int4 g32 | 4.75 | 36.85 | -0.13 | 36.21 | 36.01 | 45.01 | 27.42 | 47.38 | 8194 |
<!--END:RESULTS_MIXED-->

But the prediction that this would be the *best* use of those bits is **wrong**, and the comparison is
close enough to be worth stating flatly: at essentially the same budget, a finer group size does
slightly better.

| budget | protect the heads | shrink the groups | winner |
|---|---|---|---|
| ~3.36 bits/weight | int3 g128 + int8 heads → **34.39** | int3 g64 → **34.74** | group size, by 0.35 |
| ~4.35 bits/weight | int4 g128 + int8 heads → **36.62** | int4 g64 → **36.68** | group size, by 0.06 |

The two levers **compose**, though: adding int8 heads on top of g64 at 3 bits buys a further +0.44
(34.74 → 35.18) for another 0.16 bits/weight. So mixed precision is a real tool, just not the first
one to reach for.

### The frontier

Ranking every uniform and mixed config by what it actually costs to store:

<!--BEGIN:RESULTS_FRONTIER-->
| bits/weight | config | gIoU | on frontier? |
|--:|---|--:|---|
| 2.19 | int2 g128 | 0.35 | **yes** |
| 3.19 | int3 g128 | 33.02 | **yes** |
| 3.35 | int3 g128 + int8 heads | 34.39 | **yes** |
| 3.38 | int3 g64 | 34.74 | **yes** |
| 3.54 | int3 g64 + int8 heads | 35.18 | **yes** |
| 3.75 | int3 g32 | 35.82 | **yes** |
| 4.02 | int4 per-channel | 35.57 | dominated |
| 4.19 | int4 g128 | 36.57 | **yes** |
| 4.32 | int4 g128 + int8 heads | 36.62 | **yes** |
| 4.38 | int4 g64 | 36.68 | **yes** |
| 4.75 | int4 g32 | 36.85 | **yes** |
| 5.02 | int5 per-channel | 36.78 | dominated |
| 5.19 | int5 g128 | 36.94 | **yes** |
| 6.19 | int6 g128 | 36.96 | **yes** |
| 8.02 | int8 per-channel | 37.00 | **yes** |
| 8.19 | int8 g128 | 36.97 | dominated |
<!--END:RESULTS_FRONTIER-->

The two results worth carrying away:

- **Per-channel is Pareto-dominated at every width below 8 bits.** int4 per-channel (4.02
  bits/weight, 35.57) is beaten by **int3 g32** (3.75 bits/weight, **35.82**) — a 3-bit model with
  fine scale groups is both *smaller and better* than a 4-bit model with one scale per row. Likewise
  int5 per-channel (5.02, 36.78) loses to int4 g32 (4.75, 36.85). Nominal bit width is a poor
  predictor of accuracy; bits/weight plus granularity is a good one.
- **At 8 bits the ordering flips**, and group-wise becomes the waste: int8 per-channel (8.02, 37.00)
  dominates int8 g128 (8.19, 36.97). Above ~6 bits every config is within noise of fp32, so the only
  thing left to optimize is the metadata.

### Fit quality for every config

`rel_err` is the relative Frobenius error of the quantized weight against the original, averaged and
maximized over the quantized tensors. `worst tensor` is where the max lands — and it lands on a
**decoder head** tensor at every width from 8 to 4 bits, which is the fit-side signature of the
sensitivity result above.

<!--BEGIN:RESULTS_ABLATION-->
| weights | bits/weight | gIoU | Δ gIoU | cIoU | IoU@50 | Obj | Part | Obj&Part | n |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| int8 per-channel | 8.02 | 37.00 | +0.02 | 36.54 | 36.00 | 45.32 | 27.59 | 47.43 | 8194 |
| int5 per-channel | 5.02 | 36.78 | -0.20 | 36.37 | 36.09 | 45.20 | 27.24 | 47.35 | 8194 |
| **int5** g128 | 5.19 | 36.94 | -0.04 | 36.50 | 36.09 | 45.15 | 27.61 | 47.30 | 8194 |
| int4 per-channel | 4.02 | 35.57 | -1.41 | 36.36 | 35.44 | 43.97 | 26.17 | 45.94 | 8194 |
| **int4** g128 | 4.19 | 36.57 | -0.41 | 36.56 | 36.20 | 44.72 | 27.15 | 47.11 | 8194 |
| int4 g64 | 4.38 | 36.68 | -0.30 | 36.69 | 35.93 | 44.74 | 27.31 | 47.18 | 8194 |
| int4 g32 | 4.75 | 36.85 | -0.13 | 36.21 | 36.01 | 45.01 | 27.42 | 47.38 | 8194 |
| int4 g128, no emb | 4.19 | 36.63 | -0.35 | 36.45 | 36.15 | 44.67 | 27.40 | 46.92 | 8194 |
| int3 g128 | 3.19 | 33.02 | -3.96 | 33.90 | 32.63 | 40.21 | 23.75 | 43.78 | 8194 |
| int3 g64 | 3.38 | 34.74 | -2.24 | 35.62 | 34.31 | 42.02 | 25.05 | 46.10 | 8194 |
| int3 g32 | 3.75 | 35.82 | -1.16 | 36.73 | 35.36 | 44.18 | 26.43 | 46.20 | 8194 |
| int3, vision backbone only | 3.19 | 35.74 | -1.24 | 35.74 | 34.87 | 43.64 | 26.46 | 46.19 | 8194 |
| int3, language backbone only | 3.19 | 35.77 | -1.21 | 34.82 | 34.37 | 43.83 | 25.88 | 47.04 | 8194 |
| int3, detection heads only | 3.19 | 35.32 | -1.66 | 36.47 | 35.20 | 43.34 | 26.43 | 45.10 | 8194 |
| int4 g128, bf16 remainder | 4.19 | 36.46 | -0.52 | 36.48 | 36.10 | 44.37 | 27.08 | 47.06 | 8194 |
| int5 g128, bf16 remainder | 5.19 | 36.83 | -0.15 | 36.27 | 35.98 | 45.06 | 27.40 | 47.34 | 8194 |
| int4 g128, from the artifact | 4.19 | 36.58 | -0.40 | 36.57 | 36.26 | 44.72 | 27.15 | 47.12 | 8194 |

Per-tensor fit quality and coverage for every row above:

| weights | single-target gIoU | multi-target gIoU | coverage | rel_err mean | rel_err max | worst tensor |
|---|--:|--:|--:|--:|--:|---|
| fp32 (baseline) | 31.19 | 40.33 | — | — | — | — |
| bf16 cast | 31.27 | 40.31 | — | — | — | — |
| **int8** g128 | 31.25 | 40.29 | 98.2 % | 0.0062 | 0.0107 | transformer.decoder.ref_point_head.layers.1.weight |
| int6 g128 | 31.05 | 40.39 | 98.2 % | 0.0248 | 0.0431 | transformer.decoder.ref_point_head.layers.1.weight |
| **int5** g128 | 31.08 | 40.34 | 98.2 % | 0.0504 | 0.0872 | transformer.decoder.ref_point_head.layers.1.weight |
| **int4** g128 | 30.80 | 39.92 | 98.2 % | 0.1041 | 0.1855 | transformer.decoder.ref_point_head.layers.1.weight |
| int3 g128 | 27.03 | 36.50 | 98.2 % | 0.2219 | 0.3072 | geometry_encoder.points_pos_enc_project.weight |
| int2 g128 | 0.22 | 0.41 | 98.2 % | 0.5098 | 0.5625 | transformer.decoder.bbox_embed.layers.0.weight |
| int8 per-channel | 31.18 | 40.37 | 98.2 % | 0.0079 | 0.0140 | transformer.decoder.layers.0.linear2.weight |
| int4 per-channel | 30.42 | 38.55 | 98.2 % | 0.1341 | 0.2268 | transformer.decoder.layers.0.linear2.weight |
| int4 g64 | 30.87 | 40.05 | 98.2 % | 0.0930 | 0.1540 | transformer.decoder.ref_point_head.layers.1.weight |
| int4 g32 | 30.80 | 40.36 | 98.2 % | 0.0815 | 0.1149 | transformer.decoder.ref_point_head.layers.1.weight |
| int4 g128, no emb | 30.89 | 39.96 | 92.2 % | 0.1041 | 0.1855 | transformer.decoder.ref_point_head.layers.1.weight |
| int3 g64 | 28.58 | 38.31 | 98.2 % | 0.1985 | 0.2673 | transformer.decoder.ref_point_head.layers.1.weight |
| int3 g32 | 30.73 | 38.77 | 98.2 % | 0.1743 | 0.2129 | transformer.decoder.ref_point_head.layers.1.weight |
| int5 per-channel | 30.90 | 40.19 | 98.2 % | 0.0651 | 0.1138 | transformer.decoder.layers.0.linear2.weight |
| int4 g128, bf16 remainder | 30.68 | 39.82 | 98.2 % | 0.1041 | 0.1855 | transformer.decoder.ref_point_head.layers.1.weight |
| int5 g128, bf16 remainder | 31.02 | 40.20 | 98.2 % | 0.0504 | 0.0873 | transformer.decoder.ref_point_head.layers.1.weight |
| int4 g128, from the artifact | 30.78 | 39.94 | 98.2 % | — | — | — |
| int3, vision backbone only | 29.63 | 39.29 | 52.9 % | 0.2244 | 0.2578 | vision_trunk.blocks.0.mlp.fc1.weight |
| int3, language backbone only | 29.18 | 39.58 | 42.0 % | 0.2148 | 0.2253 | language_encoder.transformer.resblocks.22.mlp.c_proj.weight |
| int3, detection heads only | 30.48 | 38.13 | 3.3 % | 0.2252 | 0.3072 | geometry_encoder.points_pos_enc_project.weight |
| int4 g128 + **int8 heads** | 30.27 | 40.31 | 98.2 % | 0.1028 | 0.1220 | vision_trunk.blocks.0.mlp.fc1.weight |
| int3 g128 + **int8 heads** | 27.92 | 38.15 | 98.2 % | 0.2202 | 0.2578 | vision_trunk.blocks.0.mlp.fc1.weight |
| int3 g64 + **int8 heads** | 28.60 | 38.99 | 98.2 % | 0.1976 | 0.2265 | vision_trunk.blocks.0.mlp.fc1.weight |
<!--END:RESULTS_ABLATION-->
## 7. How it fails: the presence head goes quiet

The rows carry per-question intersection and union counts, and GT area is a fixed property of the
benchmark, so `pred_area = union + inter − gt_area` recovers the predicted mask area for every config
already evaluated — no extra inference. That separates the two ways this model can lose gIoU: it can
predict sloppy masks, or it can decline to predict at all.

<!--BEGIN:RESULTS_AREA-->
| weights | mean pred area / GT area | empty preds | pred>2x GT | mean IoU |
|---|--:|--:|--:|--:|
| fp32 (baseline) | 1.752 | 6.9 % | 43.7 % | 36.98 |
| bf16 cast | 1.754 | 6.9 % | 43.6 % | 36.99 |
| int8 g128 | 1.753 | 6.9 % | 43.7 % | 36.97 |
| int6 g128 | 1.742 | 6.7 % | 44.0 % | 36.96 |
| int5 g128 | 1.784 | 6.7 % | 44.3 % | 36.94 |
| int4 g128 | 1.669 | 9.4 % | 40.5 % | 36.57 |
| int3 g128 | 1.895 | 14.3 % | 41.6 % | 33.02 |
| int2 g128 | 0.051 | 98.7 % | 0.5 % | 0.35 |
| int3, vision backbone only | 1.797 | 7.7 % | 44.2 % | 35.74 |
| int3, language backbone only | 1.938 | 5.3 % | 48.7 % | 35.77 |
| int3, detection heads only | 1.660 | 15.3 % | 36.1 % | 35.32 |
| int3 g128 + int8 heads | 2.010 | 6.2 % | 49.4 % | 34.39 |
| int4 g128 + int8 heads | 1.768 | 7.0 % | 44.2 % | 36.62 |
| int8 per-channel | 1.749 | 7.0 % | 43.7 % | 37.00 |
| int5 per-channel | 1.799 | 7.3 % | 43.3 % | 36.78 |
| int4 per-channel | 1.545 | 12.4 % | 38.0 % | 35.57 |
| int4 g64 | 1.689 | 8.1 % | 42.2 % | 36.68 |
| int4 g32 | 1.758 | 6.7 % | 44.1 % | 36.85 |
| int3 g64 | 1.789 | 12.4 % | 41.8 % | 34.74 |
| int3 g32 | 1.581 | 10.6 % | 40.2 % | 35.82 |
| int4 g128, no emb | 1.684 | 9.2 % | 40.6 % | 36.63 |
| int4 g128, bf16 remainder | 1.672 | 9.4 % | 40.6 % | 36.46 |
| int5 g128, bf16 remainder | 1.786 | 6.9 % | 44.2 % | 36.83 |
| int4 g128, from the artifact | 1.672 | 9.4 % | 40.4 % | 36.58 |
| int3 g64 + int8 heads | 1.933 | 6.7 % | 47.7 % | 35.18 |
<!--END:RESULTS_AREA-->

**It declines.** The area-per-GT-area ratio is flat at ~1.75 from fp32 through int5, and the fraction
of questions where the model returns *no mask at all* is flat at 6.7–6.9 %. At int4 that jumps to
**9.4 %**, at int3 to **14.3 %**, and at int2 to **98.7 %** — the 2-bit model is not producing
garbage masks, it is producing nothing. Quantization damage in SAM3 arrives as **presence-head
silence**, not as mask degradation.

That resolves the cIoU/gIoU divergence in §5. cIoU is one global area-weighted ratio, so it is
dominated by the large masks the model still fires on and is nearly blind to a question dropping out.
gIoU weights every question equally, so each newly-silent question costs it a full unit. Hence int4
holding cIoU at 36.56 while shedding 0.41 gIoU.

**And the silence is a heads effect specifically**, which ties this section to §6. Splitting the 3-bit
damage by subsystem:

| int3 applied to | empty preds | pred area / GT | gIoU |
|---|--:|--:|--:|
| — (fp32 baseline) | 6.9 % | 1.752 | 36.98 |
| vision backbone only | 7.7 % | 1.797 | 35.74 |
| language backbone only | **5.3 %** | 1.938 | 35.77 |
| **detection heads only** | **15.3 %** | 1.660 | 35.32 |
| everything | 14.3 % | 1.895 | 33.02 |
| everything **except** the heads (int8) | **6.2 %** | 2.010 | 34.39 |

Quantizing the heads *alone* produces **more** silence (15.3 %) than quantizing the whole model
(14.3 %), while quantizing the language backbone alone makes the model slightly **more** eager than
fp32 (5.3 % vs 6.9 %). Holding the heads at 8 bits restores the empty-prediction rate to baseline
(6.2 %). So the presence-head silence in §5 comes from the heads, full stop — which is exactly why
they are the precision bottleneck of §6, and it is a satisfying closure between a
weights-side measurement and a behaviour-side one.

It also shows why restoring presence is **not sufficient**: int3 + int8 heads has baseline silence
(6.2 %) and still only scores 34.39, because the masks it does emit are now much too large (area
ratio 2.010 against fp32's 1.752). Fix the head and the body's error surfaces instead.

Finally, this is the **opposite sign to the other failure mode this project has documented**. The main
[`README.md`](README.md) reports the SA-Co/Gold regression as pure **over**-triggering: recall
unchanged, precision down, image-level FPR nearly tripled — the fine-tune became too eager to assert
presence. Quantization pushes the same head the other way: it becomes too reluctant. Same component,
opposite direction, and neither is visible in mask quality. That also makes a concrete, testable
prediction the limitations section picks up: on SA-Co/Gold, where the fp32 fine-tune over-triggers,
4-bit quantization should *help*.

## 8. Checkpoint size

Measured file sizes, against the real 3,372 MB fp32 trainer checkpoint (not a computed ideal).

<!--BEGIN:RESULTS_SIZE-->
| artifact | bits | group | codes | scales+zeros | fp32 remainder | file | vs fp32 | gIoU |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `mmr_sam3_int8_g128.pt` | 8 | 128 | 826 MB | 19 MB | 70 MB | **915 MB** | **3.68×** | 36.97 |
| `mmr_sam3_int6_g128.pt` | 6 | 128 | 619 MB | 19 MB | 70 MB | **709 MB** | **4.76×** | 36.96 |
| `mmr_sam3_int5_g128.pt` | 5 | 128 | 516 MB | 19 MB | 70 MB | **605 MB** | **5.57×** | 36.94 |
| `mmr_sam3_int4_g128.pt` | 4 | 128 | 413 MB | 19 MB | 70 MB | **502 MB** | **6.71×** | 36.57 |
| `mmr_sam3_int4_g64.pt` | 4 | 64 | 413 MB | 39 MB | 70 MB | **522 MB** | **6.46×** | 36.68 |
| `mmr_sam3_int3_g128.pt` | 3 | 128 | 310 MB | 19 MB | 70 MB | **399 MB** | **8.45×** | 33.02 |
| `mmr_sam3_int2_g128.pt` | 2 | 128 | 206 MB | 19 MB | 70 MB | **296 MB** | **11.40×** | 0.35 |
<!--END:RESULTS_SIZE-->

Two things this table makes explicit:

- **The scale/zero overhead is small but the fp32 remainder is not.** At g128 the per-group metadata
  is 19 MB, under 4 % of the int4 file. The **70 MB of unquantized fp32** — the 1.8 % of parameters
  from §2 — is 14 % of it, the single largest non-code component. §6 measures what rounding it to
  bf16 costs: **−0.11 gIoU at both int4 and int5**, to save 35 MB. Not taken; the artifacts keep the
  remainder in fp32. Worth noting how disproportionate that is — 15M params of LayerNorm and bias at
  bf16 costs as much accuracy as taking all 825M from 4 bits to 4.38.
- **Compression is sub-linear in bits** because of those two fixed costs: 8→4 bits halves the codes
  (826→413 MB) but only takes the file from 915 to 502 MB (1.82×, not 2×).

## 9. Latency and VRAM

Measured on one RTX 3090 under the same rules as mmr.md §9 — warmup excludes load and autotune,
`torch.cuda.synchronize()` brackets every timed region, images are decoded into RAM up front,
amortized at MMR val's real 3.408 questions/image. Means over **36 image encodes and 120 prompt
forwards** per dense config; the packed rows use 4 and 12, since each call is ~2× slower and the
config is load-dominated. `weights_MB` is `memory_allocated()` immediately after the model lands on
the GPU and before any activation exists; `peak_MB` is the max over the timed region at 1008×1008.

<!--BEGIN:RESULTS_LATENCY-->
| runtime | image encode (mean) | per-prompt fwd (mean) | full fwd, 1 img + 1 prompt | amortized / prediction | weights VRAM | peak VRAM |
|---|--:|--:|--:|--:|--:|--:|
| fp32 (baseline) | 111.4 ms | 67.4 ms | 178.8 ms | **100.1 ms** | 3575 MB | 5335 MB |
| int8 g128, dense | 112.0 ms | 65.8 ms | 177.8 ms | **98.7 ms** | 3575 MB | 5335 MB |
| int6 g128, dense | 111.5 ms | 63.8 ms | 175.3 ms | **96.5 ms** | 3575 MB | 5335 MB |
| int5 g128, dense | 111.2 ms | 66.2 ms | 177.4 ms | **98.8 ms** | 3575 MB | 5335 MB |
| int4 g128, dense | 111.6 ms | 66.5 ms | 178.1 ms | **99.2 ms** | 3575 MB | 5335 MB |
| int4 g32, dense | 111.6 ms | 64.3 ms | 175.9 ms | **97.1 ms** | 3575 MB | 5335 MB |
| int3 g128, dense | 111.5 ms | 66.0 ms | 177.6 ms | **98.8 ms** | 3575 MB | 5335 MB |
| int2 g128, dense | 111.1 ms | 64.4 ms | 175.5 ms | **97.0 ms** | 3575 MB | 5335 MB |
| int4 artifact, dense | 112.0 ms | 63.5 ms | 175.5 ms | **96.3 ms** | 3575 MB | 5335 MB |
| int8 g128, **packed** | 289.8 ms | 209.3 ms | 499.2 ms | **294.4 ms** | 1522 MB | 2200 MB |
| int6 g128, **packed** | 246.3 ms | 181.6 ms | 428.0 ms | **253.9 ms** | 1345 MB | 2023 MB |
| int5 g128, **packed** | 225.8 ms | 163.9 ms | 389.7 ms | **230.2 ms** | 1270 MB | 1948 MB |
| int4 g128, **packed** | 205.8 ms | 153.1 ms | 358.9 ms | **213.5 ms** | 1172 MB | 1850 MB |
| int3 g128, **packed** | 186.5 ms | 142.7 ms | 329.2 ms | **197.4 ms** | 1089 MB | 1765 MB |
| int2 g128, **packed** | 166.4 ms | 131.4 ms | 297.8 ms | **180.2 ms** | 999 MB | 1676 MB |
| int4 artifact, **packed** | 205.7 ms | 153.7 ms | 359.4 ms | **214.0 ms** | 1172 MB | 1850 MB |
<!--END:RESULTS_LATENCY-->

**How much of this is noise.** `int4 g128, packed` and `int4 artifact, packed` are the *same
configuration measured twice* (it was queued under two names). They differ by **0.5 ms in 214, or
0.2 %**, which is the error bar on this whole table. The fp32 row also reproduces mmr.md §9's
SAM3-on-question figures (111.3 ms image encode, 67.0 ms per-prompt, 99.7 ms amortized) to within
0.5 %, so the harness is measuring the same thing the earlier document did.

**`dense` is exactly neutral at every width, by construction.** It unpacks to ordinary fp32 weights
at load, so by inference time the compute graph is byte-identical to fp32 — same shapes, same
kernels, same memory traffic. The measurement agrees: image-encode time is **111.1–112.0 ms across
all nine dense rows** (a 0.8 % band spanning fp32 through int2), and the amortized spread of
96.5–100.1 ms is unordered in bit width — int6 appears "faster" than int5 and int4, which is
physically meaningless. Every accuracy number in this document was measured in this mode. **A
smaller checkpoint on disk is the entire benefit; there is no free speedup.**

**`packed` trades speed for VRAM, and the trade scales with bit width** — this is the one place the
width does move the clock, because `unpack_bits` runs one shift-and-or pass per bit inside every
forward:

| packed width | amortized | vs fp32 | Δ per bit | weights VRAM | vs fp32 |
|---|--:|--:|--:|--:|--:|
| int8 | 294.4 ms | 2.94× slower | — | 1522 MB | 2.35× less |
| int6 | 253.9 ms | 2.54× | −20.3 ms | 1345 MB | 2.66× |
| int5 | 230.2 ms | 2.30× | −23.7 ms | 1270 MB | 2.81× |
| int4 | 213.5 ms | 2.13× | −16.7 ms | 1172 MB | 3.05× |
| int3 | 197.4 ms | 1.97× | −16.1 ms | 1089 MB | 3.28× |
| int2 | 180.2 ms | 1.80× | −17.2 ms | 999 MB | 3.58× |

Fewer bits are both smaller *and* faster here, which is the opposite of the usual quantization
trade-off and is purely an artifact of the unpacking loop: it costs a fixed ~17 ms per bit-plane
pass, so dropping a bit drops a pass. (int2 is included for the trend only — at 0.35 gIoU that model
is useless; §5.)

**fp32 has no packed row and cannot have one.** Packed means holding integer codes resident and
decoding them in the forward pass; fp32 weights are not encoded, so there is nothing to pack. It is
a dense-only baseline by definition. On a 24 GB card none of this is worth it;
it matters only where the model must fit in ~2 GB, or where many models share one device.

**Why 1,172 MB at int4 and not ~500 MB.** `packed` only wraps `nn.Linear`, so 285 of the 348
quantized tensors stay packed and the rest fall back to dense fp32 — the 61
`MultiheadAttention.in_proj_weight` tensors (82.8M params, 331 MB) and the CLIP token embedding
(50.6M, 202 MB) from §2, plus the 70 MB unquantized remainder. Wrapping those two families would
bring the resident footprint down to roughly the artifact's own 502 MB. That is the obvious next step
and it is not done here.

Turning k-bit storage into k-bit *inference* — where fewer bits are faster because the matmul itself
is cheaper, not because there is less unpacking to do — needs a fused dequant-matmul kernel. This
work does not attempt one.

## 10. Limitations

- **RTN only.** No GPTQ, AWQ, SmoothQuant or any calibration-based method. Those exist precisely to
  buy back what RTN loses at 4 bits and below, so the int4/int3 rows here are a **floor**, not a
  verdict on 4-bit SAM3. The 0.41 gIoU at int4 is what you get for free; a calibrated method would
  likely close it.
- **Weights only.** Activations stay in bf16 autocast, exactly as the fp32 baseline runs them. This
  is a checkpoint-size result; an activation-quantized W8A8 path is a different measurement and would
  need different kernels.
- **No fused kernel, so no speed win.** The `dense` mode that produced every accuracy number holds
  fp32 weights at runtime and is measured latency- and VRAM-neutral at every width from 8 to 2 bits
  (§9). `packed` saves real VRAM (up to 3.28× at int3) but costs 2.0–2.9× the latency in pure
  PyTorch. Turning k-bit storage into k-bit *inference* needs a fused dequant-matmul, not attempted
  here.
- **MMR val only.** Same caveat mmr.md §11 carries: this is the split the model was tuned against,
  and no result here is a held-out test number. Whether quantization sensitivity looks the same on
  SA-Co/Gold — where the fp32 fine-tune already regresses through over-triggering — is untested, and
  §7 gives a specific reason to expect it to differ: quantization pushes the presence head toward
  silence, which is the direction that *helps* an over-triggering model. That is a real experiment
  this document does not run.
- **One checkpoint.** Everything here is the MMR-trained model. The SA-1B reasonseg fine-tune and
  stock `facebook/sam3` are not re-measured, so "SAM3 tolerates 5-bit weights" is an inference from
  one model, not a demonstration across three.
- **`packed` mode is incomplete.** It wraps `nn.Linear` only, so the 61 `MultiheadAttention`
  in_proj weights and the CLIP token embedding — 133M params, 533 MB in fp32 — fall back to dense and
  the resident footprint is 1,172 MB rather than the ~502 MB the artifact size implies (§9).
- **Mixed precision was tested at one setting.** Heads at int8, bodies at int3/int4, group sizes
  128/64. Intermediate head widths (int5, int6), per-subsystem group sizes, and holding the
  *unquantized remainder* at higher precision instead were not tried, so §6's "group size beats head
  protection" conclusion is specific to the four points measured, not a general ordering.
- **The `min_numel` cutoff is unswept.** Tensors under 65,536 elements are left in fp32 by fiat. That
  covers 1.8 % of parameters and was never varied; the worst per-tensor relative error at every
  width lands on small decoder-head tensors (`transformer.decoder.ref_point_head.layers.1` at
  8–4 bits), which suggests the cutoff is doing useful work, but that is an observation, not a sweep.

## 11. Using a quantized checkpoint

The artifact is a single `.pt` holding packed uint8 codes, bf16 scales, uint8 zeros and the
unquantized remainder, plus the config needed to reconstruct the model:

```python
import sys; sys.path.insert(0, "/workspace/reasonseg/quant/code")
import sam3_quant_model as Q
from sam3.model.sam3_image_processor import Sam3Processor

model, info = Q.build(quant_ckpt="/mnt/data0/ameen/quant_ckpts/mmr_sam3_int5_g128.pt",
                      quant_mode="dense",     # or "packed" for low VRAM, see §9
                      device="cuda:0")
proc = Sam3Processor(model, device="cuda:0", confidence_threshold=0.5)
```

From there it is an ordinary SAM3 image model — `proc.set_image(...)` /
`proc.set_text_prompt(...)`, identical to the fp32 path. On MMR val's first question
("What could indicate that one of the individuals in the image might be shielding themselves from the
sun or rain?", GT `hat`, 70 px) that int5 artifact returns one mask at **IoU 0.8816** against fp32's
**0.8919** — the whole result in miniature.

Note the import order: `sam3_quant_model` puts `/workspace/sam3` on `sys.path` as a side effect, so it
has to be imported before `sam3.model.*`.

## 12. Reproducing

```bash
PY=/workspace/envs/sam3/bin/python
CODE=quant/code

$PY $CODE/test_wquant.py                    # quantizer correctness (§1-§3)
bash $CODE/run_quant_sweep.sh               # the bit-width curve, full MMR val (§5)
bash $CODE/run_quant_ablations.sh           # group size / embeddings / subsystem (§6)
bash $CODE/write_artifacts.sh               # the .pt files and their sizes (§8)
bash $CODE/run_quant_phase3.sh              # artifact round-trip + determinism (§4)
bash $CODE/bench_artifacts.sh               # latency and VRAM (§9)
$PY $CODE/quant_table.py                    # results tables
$PY $CODE/analyze_area.py                   # failure-mode analysis (§7)
$PY $CODE/size_table.py                     # on-disk sizes
```

Each eval config is 3-way sharded across the box's three RTX 3090s and takes ~5.5 min for the full
8,194-question val split. `CUDA_VISIBLE_DEVICES` selects the GPU and `--device` is always `cuda:0`,
because SAM3's image processor builds index tensors on the default device and will otherwise fail
with a cross-device index error.

| file | what it does |
|---|---|
| `quant/code/wquant.py` | the quantizer: group-wise RTN, bit-plane packing, `QuantLinear`, artifact I/O |
| `quant/code/sam3_quant_model.py` | builds a SAM3 image model from a trainer or quantized checkpoint |
| `quant/code/quantize_ckpt.py` | fp32 checkpoint → packed k-bit artifact |
| `quant/code/run_mmr_quant.py` | MMR val eval, same protocol as `mmrcomp/code/run_sam3_mmr.py` |
| `quant/code/merge_quant.py` | merges shards; adds exact per-subset cIoU |
| `quant/code/test_wquant.py` | the correctness suite (§1–§3) |
| `quant/code/analyze_area.py` | recovers predicted mask area from stored counts (§7) |
| `quant/code/check_determinism.py` | per-question diff of two runs of one config |
| `quant/code/bench_quant_latency.py` | per-prediction latency + resident VRAM |
| `quant/code/{quant,size}_table.py` | the tables in this document |

Metric dumps are in `quant/eval/mmr_*.json`; artifacts in `/mnt/data0/ameen/quant_ckpts/`.
