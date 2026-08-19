#!/bin/bash
# Incremental data-scaling: train one SAM3 model through 4 chained stages (1k/2k/4k/8k cumulative),
# benchmarking each finished stage in the background (GPU1+2) while the next stage trains (GPU0).
set -u
SAM3=/workspace/newsam/sam3
RUNS=/mnt/data0/ameen/mmr_runs
SCR=/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad
export HF_HOME=/mnt/data0/ameen/hf_cache PYTHONPATH=/workspace/newsam/sam3
for stage in 1 2 3 4; do
  echo "===== TRAIN stage $stage  $(date) ====="
  cd $SAM3
  CUDA_VISIBLE_DEVICES=0 /workspace/envs/sam3/bin/python sam3/train/train.py \
    -c configs/newsam/mmr_stage${stage}.yaml --use-cluster 0 --num-gpus 1 \
    > $RUNS/train_stage${stage}.log 2>&1
  ckpt=$RUNS/mmr_stage${stage}/checkpoints/checkpoint.pt
  if [ ! -f "$ckpt" ]; then echo "!!! STAGE $stage FAILED (no checkpoint) — see $RUNS/train_stage${stage}.log"; exit 1; fi
  echo "===== stage $stage trained; launching benchmark in bg  $(date) ====="
  bash $SCR/bench_stage.sh $stage "$ckpt" > $RUNS/bench_stage${stage}.log 2>&1 &
done
wait   # wait for the last benchmark(s)
echo "===== ALL STAGES DONE  $(date) ====="
