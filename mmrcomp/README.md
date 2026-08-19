# MMR benchmark: M²SA-7B vs. SAM3 (base & reasonseg fine-tune)

Head-to-head on the **MMR** benchmark (*Multi-target and Multi-granularity Reasoning
Segmentation*, ICLR 2025 — [jdg900/MMR](https://github.com/jdg900/MMR)), scored on the
official **val** split: **2,404 images / 8,194 (image, question) pairs**, PACO-LVIS masks.

MMR's task: given an image and an *implicit reasoning question* (e.g. "What might be the
purpose of the item holding the flower?"), output the pixel mask(s) of the answer. Targets
can be multiple objects **and** their parts (multi-target, multi-granularity).

## Models compared
- **M²SA-7B** — MMR's own model (LLaVA-7B + SAM ViT-H, LISA-style). Built for this task.
  Downloaded from [`jdg900/M2SA-7B`](https://huggingface.co/jdg900/M2SA-7B).
- **LISA-7B** — the benchmark's main baseline and the model M²SA forks, from
  [`xinlai/LISA-7B-v1`](https://huggingface.co/xinlai/LISA-7B-v1).
- **PixelLM-7B** — CVPR 2024 multi-target reasoning segmentation (codebook + own mask decoder,
  no SAM), from [`maverickrzw/PixelLM-7B`](https://huggingface.co/maverickrzw/PixelLM-7B).
  Zero-shot on MMR, like LISA.
- **SAM3 base** — `facebook/sam3` zero-shot (concept/phrase segmenter).
- **SAM3 fine-tuned** — your `sam3-reasonseg-mixed` checkpoint (SA-1B complex-referring-expression fine-tune).

## The metrics — gIoU and cIoU
Each item is one (image, query) pair; the model outputs a binary mask **P**, ground truth is
mask **G** (union of all target masks). Per pair, IoU = |P∩G| / |P∪G|.

- **gIoU** (mean IoU) = `(1/N) Σ IoUᵢ` — every pair weighted **equally** (a tiny part counts as
  much as a big object). Convention: IoU=1 when both P and G are empty.
- **cIoU** (cumulative IoU) = `Σ|Pᵢ∩Gᵢ| / Σ|Pᵢ∪Gᵢ|` — one global ratio, **area-weighted**
  (large masks dominate).

Reporting both is diagnostic: a model that only nails big objects looks good on cIoU but bad on
gIoU. MMR's official code computes them **per-target** with the `[SEG]` count teacher-forced; to
score SAM3's unordered mask set the same way, the cross-model comparison uses a **per-question
union** mask (identical to MMR's for single-target questions).

## Results (MMR val, gIoU / cIoU, ×100)

| Model | input | gIoU | cIoU | notes |
|---|---|--:|--:|---|
| **M²SA-7B** | question (teacher-forced, MMR official) | **36.0** | **54.7** | native per-target **28.1 / 49.0** — reproduces paper's **27.8 / 48.6** ✓ |
| M²SA-7B | question (free-generation) | 25.2 | 30.1 | honest question-only (no answer leak) — the fair input to compare with SAM3 |
| LISA-7B | question (teacher-forced) | 12.5 | 19.2 | our run; per-target **8.94** vs paper's 13.8 — does not reproduce, see [`mmr.md`](../../mmr.md) |
| LISA-7B | question (free-generation) | 14.4 | 18.9 | teacher forcing *hurts* LISA (single-`[SEG]` model forced to emit N) |
| PixelLM-7B | question (free-generation) | 19.2 | 21.8 | zero-shot; multi-target design shows as single→multi **+3.9** vs LISA's +0.6 |
| SAM3 **base** | question | 0.6 | 1.5 | can't parse a reasoning question |
| SAM3 **fine-tuned** | question | **11.1** | 10.6 | **~18× over base** — the complex-referring training transferred |
| SAM3 base | *oracle concept* | **38.8** | 50.8 | given the target name — **beats M²SA on masks** |
| SAM3 fine-tuned | *oracle concept* | 30.1 | 45.2 | fine-tune *regresses* on plain concepts (over-triggers) |
| **SAM3 + MMR train** | question | **37.2** | 36.9 | trained on MMR itself — see [`scaling_ladder.md`](scaling_ladder.md); **beats M²SA's teacher-forced 36.0 with no answer leak** |

By granularity (gIoU): M²SA obj 39.2 / part 30.2 / obj&part 43.4 · SAM3-ft-question obj 11.8 /
part 8.4 / obj&part 14.9 · SAM3-base-oracle obj 46.6 / part 27.5 / obj&part 52.5.

### Which is better?
**Off the shelf, M²SA-7B wins decisively** — it has an LLM that interprets the implicit question;
SAM3 alone essentially cannot (0.6 gIoU). **Trained on MMR, SAM3 wins** (37.2 vs 36.0 teacher-forced,
vs 25.2 question-only) — see [Training SAM3 on MMR](#training-sam3-on-mmr). Qualifications:

0. **Apples-to-apples (both get only the question):** M²SA's official 36.0 is *teacher-forced* on the
   GT answer text, which names the targets. In honest free-generation M²SA scores **25.2 gIoU** — an
   ~11-point drop — so teacher-forcing is a real advantage. The fair question-only line is
   **M²SA 25.2 vs your SAM3 fine-tune 11.1** (vs SAM3 base 0.6): M²SA still wins, by less than the
   headline implies.
1. **Your fine-tune closes most of the base gap** on questions (0.6 → 11.1 gIoU, ~18×), though still
   ~⅓ of M²SA.
2. **The gap is language, not pixels.** Handed the target *concept*, SAM3-base (38.8) **out-segments
   M²SA** (36.0). SAM3 is the better segmenter; M²SA is the better reasoner. **Confirmed:** training
   SAM3 on MMR reaches **37.2 from the raw question**, within 1.65 of its own oracle-concept ceiling
   — the language gap closes with in-distribution data, no LLM needed.
3. **Fine-tune trade-off** (matches your README's SA-Co/Gold finding): it *helps* the complex-question
   distribution (0.6→11.1) but *hurts* the plain-concept distribution (38.8→30.1, over-triggering).

## Training SAM3 on MMR
[`scaling_ladder.md`](scaling_ladder.md) trains the stock SAM3 checkpoint on MMR train itself, with
MMR-val benchmarks at 0.25 / 0.5 / 1 / 2 / 4 / 8 / 16 / 32 / 64 / 100 % of the data (one continuous
run, one optimizer, no per-rung checkpoints). Scaling is **monotonic to 100 %**, ending at
**37.15 gIoU** — above M²SA-7B's teacher-forced 36.0, and within 1.65 of what base SAM3 scores when
handed the oracle concept (38.8). It confirms point 2 above: the gap was language, not pixels, and
in-distribution data closes it without an LLM. Parts remain the ceiling (27.8 vs 45.6 for objects).

## Inference cost
Per-prediction latency for the same models is in [`latency.md`](latency.md): on one RTX 3090,
SAM3 costs **~100 ms** per (image, question) amortized vs. M²SA's **4.2 s** free-generation
(**42× slower**) or 340 ms teacher-forced. SAM3-base-oracle beats M²SA-teacher on gIoU *and* is
2.6× faster.

## Examples (`examples/`, contact sheet `mmr_examples_contact_sheet.png`)
Each row: the **PROMPT** banner (also overlaid on the image), then
`Image | Ground truth | M²SA-7B | SAM3 base·Q | SAM3 fine-tuned·Q | SAM3 base·oracle | SAM3 fine-tuned·oracle`.
Overlay legend: **red fill = model prediction**, **green outline = ground-truth target** (a perfect
prediction fills the outline). Per-panel IoU printed below.

- `ex0/1_reasoning_gap_obj` — SAM3 can't parse the question (IoU~0) but nails it given the concept (~0.99); M²SA reasons it out.
- `ex2_finetune_helps_q` — fine-tuned SAM3 solves the question outright (0.99) where base fails (0.0).
- `ex3_sam3_oracle_beats_m2sa` — **M²SA fails (0.0)**, fine-tuned SAM3 gets it from the question (0.98).
- `ex4_multi_target` — an 8-target object&part query (union).
- `ex5_part_level` — part segmentation ("laptop logo"): SAM3-oracle 0.99, M²SA 0.99, SAM3-question 0.49.
- `ex6_hard_for_all` — everyone fails (honest failure case).

## Code (`code/`)
All scripts are self-contained; they share `mmr_common.py` (GT decode + gIoU/cIoU meters) so both
environments score the identical ground truth identically.

| file | env | what it does |
|---|---|---|
| `mmr_common.py` | both | load MMR val, decode per-target/union GT masks, `UnionMeter` (union gIoU/cIoU) + `NativeMeter` (MMR's official per-target metric), granularity split. |
| `run_m2sa_mmr.py` | m2sa (py3.8/torch2.1-cu118/tf4.31) | runs M²SA-7B. `--mode teacher` = MMR's official teacher-forced protocol (reproduces the paper + a union mask); `--mode gen` = question-only free generation. Sharded via `--shard/--nshards`. |
| `run_sam3_mmr.py` | sam3 (torch2.10) | runs a SAM3 checkpoint. `--prompt-mode question` (raw question) or `concept` (oracle target name). `--ckpt base|trained`. |
| `merge_mmr.py` | any | merges shard outputs → final gIoU/cIoU, by-granularity, single/multi-target. |

### Reproduce
```bash
# M2SA (from the MMR repo dir, m2sa env), 3-GPU shards, official protocol:
for S in 0 1 2; do CUDA_VISIBLE_DEVICES=$S HF_HOME=/mnt/data0/ameen/hf_cache \
  python run_m2sa_mmr.py --mode teacher --device cuda:0 --nshards 3 --shard $S \
  --out m2sa_teacher_s$S.json & done; wait
python merge_mmr.py m2sa_teacher "m2sa_teacher_s*.json"

# SAM3 fine-tuned, question prompt:
for S in 0 1 2; do CUDA_VISIBLE_DEVICES=$S python run_sam3_mmr.py \
  --ckpt trained --prompt-mode question --device cuda:0 --nshards 3 --shard $S \
  --out sam3_trained_question_s$S.json & done; wait
python merge_mmr.py sam3_trained_question "sam3_trained_question_s*.json"
```

Full per-run metric dumps (overall / by-granularity / single-vs-multi) are in
`../mmr_eval/*.json`.
