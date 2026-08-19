#!/bin/bash
# Run all 4 SAM3 configs (base/trained x question/concept), each sharded 3x across GPUs.
# Configs run sequentially; 3 shards run in parallel within a config.
set -u
PY=/workspace/envs/sam3/bin/python
SCR=/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad
OUT=/mnt/data0/ameen/mmr_out
cd /workspace/reasonseg
for CFG in "base question" "trained question" "base concept" "trained concept"; do
  set -- $CFG; CKPT=$1; PM=$2
  echo "=== SAM3 $CKPT $PM $(date) ==="
  for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$S $PY $SCR/run_sam3_mmr.py \
      --ckpt $CKPT --prompt-mode $PM --device cuda:0 --nshards 3 --shard $S \
      --out $OUT/sam3_${CKPT}_${PM}_s$S.json \
      > $OUT/sam3_${CKPT}_${PM}_s$S.log 2>&1 &
  done
  wait
  echo "=== done SAM3 $CKPT $PM $(date) ==="
done
echo "ALL SAM3 DONE"
