#!/bin/bash
# MMR data-scaling ladder on an ORIGINAL SAM3 checkpoint.
#
# ONE continuous 1-epoch fine-tune over all 45,618 MMR train images (154,124
# image-question samples), with an MMR-val benchmark taken in-flight at
# 0.25 / 0.5 / 1 / 2 / 4 / 8 / 16 / 32 / 64 / ~100 % of the data. The weights
# just keep training across every rung: one optimizer, one LR schedule, one
# rolling checkpoint overwritten in place -- no per-rung checkpoint copies.
#
# Each rung pauses training, saves the rolling checkpoint, evaluates it 3-way
# sharded across the training GPUs, then resumes. Results land incrementally in
# /workspace/reasonseg/mmr_eval/mmrscale_*.json, so the curve is usable at
# any point and the run can be killed whenever.
set -u
SAM3=/workspace/newsam/sam3
RUNS=/mnt/data0/ameen/mmr_runs/mmr_scale
CODE=/workspace/reasonseg/mmrcomp/code
mkdir -p "$RUNS" /mnt/data0/ameen/mmr_out/scale

export HF_HOME=/mnt/data0/ameen/hf_cache
export PYTHONPATH=/workspace/newsam/sam3
# Model-only rolling checkpoint: 3.4 GB instead of 10 GB (drops optimizer/scaler
# state). Disk is at ~20 GB free and we never resume mid-run.
export NEWSAM_MODEL_ONLY_CKPT=1
# Guards added after run 1 silently stopped learning (NaN grads from the fused
# Triton focal kernel): abort if >50% of the last 200 steps get skipped, and
# report where any NaN gradient came from the first few times it happens.
export NEWSAM_MAX_SKIP_RATE=0.5
export NEWSAM_SKIP_WINDOW=200
export NEWSAM_GRAD_DIAG_REPORTS=3
# The ladder. 1.0 is covered by the end-of-epoch bench (tag "epoch_1"); 0.9995
# is a belt-and-braces near-100 % rung taken through the same barriered path.
export NEWSAM_BENCH_FRACS="0.0025,0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.9995"
export NEWSAM_BENCH_CMD="bash $CODE/bench_scale.sh {ckpt} {tag}"

# SINGLE GPU on purpose. Measured on this box (3x RTX 3090, PHB / no P2P):
#   grouped  1 GPU  -> 1.10 s/step over 3.38 samples = 0.33 s/sample
#   grouped  3 GPUs -> 17.0 s/step over 3.38 samples = 5.03 s/sample
# DDP gradient all-reduce of ~843M params over PCIe without P2P costs ~16 s/step,
# so 3 GPUs are 15x SLOWER per sample than one. One GPU trains; the other two are
# free, and each rung's benchmark shards across all three while training is paused.
cd "$SAM3"
exec /workspace/envs/sam3/bin/python sam3/train/train.py \
  -c configs/newsam/mmr_scale.yaml --use-cluster 0 --num-gpus 1
