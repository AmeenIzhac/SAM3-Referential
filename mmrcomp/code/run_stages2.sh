#!/bin/bash
# Stages 2-4: continue the chained model. After each stage, write a detector-prefixed
# "chain" checkpoint that the next stage's checkpoint_path can actually load.
set -u
SAM3=/workspace/newsam/sam3
RUNS=/mnt/data0/ameen/mmr_runs
SCR=/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad
export HF_HOME=/mnt/data0/ameen/hf_cache PYTHONPATH=/workspace/newsam/sam3
for stage in 2 3 4; do
  echo "===== TRAIN stage $stage  $(date) ====="
  cd $SAM3
  CUDA_VISIBLE_DEVICES=0 /workspace/envs/sam3/bin/python sam3/train/train.py \
    -c configs/newsam/mmr_stage${stage}.yaml --use-cluster 0 --num-gpus 1 \
    > $RUNS/train_stage${stage}.log 2>&1
  ckpt=$RUNS/mmr_stage${stage}/checkpoints/checkpoint.pt
  if [ ! -f "$ckpt" ]; then echo "!!! STAGE $stage FAILED (no checkpoint)"; exit 1; fi
  # make chain checkpoint (detector.-prefixed) for the next stage
  /workspace/envs/sam3/bin/python -c "
import torch
ck=torch.load('$ckpt',map_location='cpu',weights_only=False)
torch.save({'model':{('detector.'+k):v for k,v in ck['model'].items()}}, '$RUNS/mmr_stage${stage}/checkpoints/checkpoint_chain.pt')
print('chain written for stage $stage')"
  echo "===== stage $stage trained; benchmarking in bg  $(date) ====="
  bash $SCR/bench_stage.sh $stage "$ckpt" > $RUNS/bench_stage${stage}.log 2>&1 &
done
wait
echo "===== STAGES 2-4 DONE  $(date) ====="
