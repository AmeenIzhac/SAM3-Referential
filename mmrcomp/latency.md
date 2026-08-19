# Inference latency: M²SA-7B vs. SAM3

Per-prediction wall time for the same models compared in [`README.md`](README.md), measured on
**real MMR val (image, question) pairs**. One "prediction" = one (image, question) pair → one
predicted mask.

## How it was measured (fairness rules)

The goal is the **steady-state per-prediction cost you'd see running many**, not a cold one-shot
timing. So:

- **Model load, CUDA context init, cuDNN autotune, weight paging are excluded** — 10 warmup
  iterations (5 for M²SA) run before the clock starts.
- **`torch.cuda.synchronize()` brackets every timed region**, so what's measured is completed GPU
  work, not async kernel-launch return.
- **No disk I/O inside the timed region** — all images are decoded and resident in RAM before timing.
- **Constant overhead is separated out, not folded in**: mask→CPU transfer and CPU-side
  preprocessing are timed as their own line items (both turn out to be negligible: < 0.5 ms and
  ~28 ms/image respectively).
- Same GPU (**one RTX 3090**, exclusive, runs serialized so no contention), same dtype (**bf16**),
  same 40 images / 100–200 timed calls per config.
- Reported as **median** (mean/p90/min in the JSON dumps). For SAM3 concept mode the **mean** is
  used, because that mode issues a variable number of prompts per question (1.5 on average).

### The one structural difference that matters

The two models divide work differently, and folding this in wrongly is the easy way to get an
unfair ratio:

- **SAM3** encodes the image **once per image** (`set_image`, 111 ms) and then runs the text encoder
  + grounding decoder **once per prompt** (`set_text_prompt`, ~66 ms). The image encoding is reused
  across every question on that image.
- **M²SA** recomputes its vision towers (CLIP + SAM ViT-H) **inside every forward**. There is
  nothing to amortize: one question = one full forward.

So SAM3 is reported two ways: **cold** (one image, one question — nothing reused) and **amortized**
(at MMR val's real **3.408 questions/image**, i.e. 8,194 q / 2,404 images). The amortized number is
the honest "running many" figure; the cold number is the worst case.

## Results (RTX 3090, bf16, ms per prediction)

| Model / setting | image enc | per-prompt fwd | **cold** | **amortized** | throughput |
|---|--:|--:|--:|--:|--:|
| **M²SA-7B** — question, free-generation | — | 4220.1 | **4220.1** | **4220.1** | 0.24 /s |
| **M²SA-7B** — teacher-forced (MMR official) | — | 340.0 | **340.0** | **340.0** | 2.94 /s |
| SAM3 base — question | 111.2 | 65.8 | 177.0 | **98.4** | 10.16 /s |
| SAM3 fine-tuned — question | 111.3 | 67.0 | 178.3 | **99.7** | 10.03 /s |
| SAM3 base — oracle concept | 111.8 | 98.2 | 210.0 | **131.0** | 7.64 /s |
| SAM3 fine-tuned — oracle concept | 111.8 | 100.6 | 212.3 | **133.4** | 7.50 /s |

Excluded/negligible line items: mask→CPU 0.01–0.4 ms (both); M²SA CPU preprocessing 28.3 ms/image;
SAM3 at resolution 1008, M²SA at 1024.

**M²SA generation is highly variable** (median 4220, mean 4426, p90 6437, min 2328 ms) because it's
autoregressive — ~101 tokens generated per answer. SAM3 and M²SA-teacher are near-constant-time
(SAM3 p90 within 4 % of median; M²SA-teacher within 2 %).

## The ratios

| comparison | ratio |
|---|--:|
| M²SA free-generation ÷ SAM3 fine-tuned (question) | **42.3× slower** |
| M²SA free-generation ÷ SAM3 base (oracle concept) | **32.2× slower** |
| M²SA teacher-forced ÷ SAM3 fine-tuned (question) | 3.4× slower |
| M²SA teacher-forced ÷ SAM3 base (oracle concept) | 2.6× slower |

Note that M²SA's 340 ms teacher-forced number **is not a deployable setting** — it requires the
ground-truth answer text, so it only tells you the cost of the prefill+decode forward without
generation. The honest question-only cost is **4.2 s**.

## Reading it against accuracy

Pairing with the gIoU numbers from [`README.md`](README.md):

| Model / setting | gIoU | ms/pred | gIoU-points per second |
|---|--:|--:|--:|
| SAM3 base — oracle concept | 38.8 | 131.0 | **296.3** |
| SAM3 fine-tuned — question | 11.1 | 99.7 | 111.3 |
| M²SA-7B — teacher-forced | 36.0 | 340.0 | 105.9 |
| M²SA-7B — free-generation | 25.2 | 4220.1 | 6.0 |
| SAM3 base — question | 0.6 | 98.4 | 6.1 |

1. **The apples-to-apples comparison flips on cost.** The fair accuracy line was M²SA-gen **25.2**
   vs. SAM3 fine-tuned **11.1** — M²SA wins by ~2.3×. But it costs **42× more time per prediction**.
   Per unit of compute, SAM3 is ~18× more efficient in that pairing.
2. **SAM3-base with an oracle concept beats M²SA-teacher on both axes** — higher gIoU (38.8 vs 36.0)
   at 2.6× less time. This sharpens the README's "the gap is language, not pixels" point: the
   segmenter is both better *and* far cheaper; the expensive part is the 7B LLM doing the reasoning.
3. **This strengthens the SAM3 + query-rewriter proposal.** A question→concept rewrite adds one
   short LLM call, but it can be a much smaller model than a 7B LISA-style decoder, and — unlike
   M²SA — the rewrite is per-question text only, so it batches and caches independently of the
   image encoder.
4. **Fine-tuning is latency-neutral.** Base vs. fine-tuned differ by ~1 ms (99.7 vs 98.4); the
   accuracy changes in the README cost nothing at inference time.
5. **Amortization matters for SAM3, not M²SA.** SAM3 goes 177 → 98 ms per prediction as soon as an
   image carries more than one query (3.41 on MMR); M²SA is flat, because it re-encodes the image
   every time.

## Reproduce

```bash
SP=mmrcomp/code

# SAM3 (sam3 env), one config per invocation
CUDA_VISIBLE_DEVICES=0 /workspace/envs/sam3/bin/python $SP/bench_sam3_latency.py \
    --ckpt base --prompt-mode question --n-images 40 --n-prompts 200 --warmup 10 \
    --out mmr_eval/lat_sam3_base_question.json

# M2SA (m2sa env, cwd = MMR repo)
CUDA_VISIBLE_DEVICES=0 /mnt/data0/ameen/envs/m2sa/bin/python $SP/bench_m2sa_latency.py \
    --mode gen --n-images 40 --n-preds 100 --warmup 5 \
    --out mmr_eval/lat_m2sa_gen.json
```

Raw dumps (mean / median / p90 / min for every timed component) are in
`../mmr_eval/lat_{sam3_*,m2sa_*}.json`.
