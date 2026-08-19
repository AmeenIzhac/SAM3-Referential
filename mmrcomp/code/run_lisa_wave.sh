#!/bin/bash
# Full LISA-7B evaluation on MMR val, 3-way sharded, both protocols.
#   teacher : MMR's official teacher-forced protocol -> reproduces the paper's
#             per-target number AND yields a union mask per question
#   gen     : question-only free generation -> the honest, deployable setting
# Identical harness, prompts and meters as run_m2sa_mmr.py, so LISA / M2SA / SAM3
# are all scored the same way.
set -u
PY=/mnt/data0/ameen/envs/m2sa/bin/python
CODE=/workspace/reasonseg/mmrcomp/code
LISA=/tmp/claude-1009/-workspace-reasonseg/d6cd4b24-35e9-4fa3-8cad-354194dce19a/scratchpad/LISA
OUT=/mnt/data0/ameen/mmr_out
export HF_HOME=/mnt/data0/ameen/hf_cache
mkdir -p "$OUT"
cd "$LISA"

for mode in teacher gen; do
  echo "===== LISA $mode  $(date) ====="
  for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$S $PY "$CODE/run_lisa_mmr.py" --mode $mode \
      --device cuda:0 --nshards 3 --shard $S \
      --out "$OUT/lisa_${mode}_s$S.json" > "$OUT/lisa_${mode}_s$S.log" 2>&1 &
  done
  wait
  n=$(ls "$OUT"/lisa_${mode}_s*.json 2>/dev/null | wc -l)
  if [ "$n" -ne 3 ]; then
    echo "!!! LISA $mode FAILED: only $n/3 shards"; tail -15 "$OUT"/lisa_${mode}_s*.log; exit 1
  fi
  $PY "$CODE/merge_mmr.py" "lisa_${mode}" "$OUT/lisa_${mode}_s*.json"
done
echo "===== LISA DONE  $(date) ====="
