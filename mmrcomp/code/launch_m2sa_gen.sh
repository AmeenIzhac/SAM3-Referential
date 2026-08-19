#!/bin/bash
# M2SA free-generation (question-only) full run, 3 shards across GPUs.
set -u
PY=/mnt/data0/ameen/envs/m2sa/bin/python
SCR=/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad
OUT=/mnt/data0/ameen/mmr_out
cd /tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad/MMR
for S in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$S HF_HOME=/mnt/data0/ameen/hf_cache \
  $PY $SCR/run_m2sa_mmr.py --mode gen --device cuda:0 --nshards 3 --shard $S \
    --out $OUT/m2sa_gen_s$S.json > $OUT/m2sa_gen_s$S.log 2>&1 &
done
wait
echo "ALL M2SA GEN DONE"
