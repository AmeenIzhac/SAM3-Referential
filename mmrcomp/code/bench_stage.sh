#!/bin/bash
# Benchmark one stage checkpoint on MMR val (question prompt), 2-way sharded on GPU1+GPU2, then merge.
set -u
stage=$1; ckpt=$2
PY=/workspace/envs/sam3/bin/python
SCR=/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad
OUT=/mnt/data0/ameen/mmr_out
cd /workspace/reasonseg
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=$((S+1)) $PY $SCR/run_sam3_mmr.py --ckpt-path "$ckpt" --prompt-mode question \
    --device cuda:0 --nshards 2 --shard $S --out $OUT/mmr_stage${stage}_q_s$S.json \
    > $OUT/mmr_stage${stage}_q_s$S.log 2>&1 &
done
wait
$PY $SCR/merge_mmr.py mmr_stage${stage} "$OUT/mmr_stage${stage}_q_s*.json"
echo "BENCH stage $stage merged"
