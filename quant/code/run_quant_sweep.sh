#!/bin/bash
# Full MMR-val sweep over weight precisions for the MMR-trained SAM3 checkpoint.
#
# One config at a time, each 3-way sharded across the box's 3 RTX 3090s (the
# same layout bench_scale.sh used during training: CUDA_VISIBLE_DEVICES picks
# the GPU, --device is always cuda:0, because SAM3's image processor builds
# index tensors on the default device).
#
# Usage: run_quant_sweep.sh [LIMIT]     # LIMIT = images per shard, for smoke runs
set -u
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/quant/code
OUT=/mnt/data0/ameen/quant_out
mkdir -p "$OUT" /workspace/reasonseg/quant/eval
LIM=""
[ $# -ge 1 ] && LIM="--limit $1"

# tag:flags -- flags are passed through to run_mmr_quant.py verbatim
CONFIGS=(
  "fp32:"
  "bf16:--weight-dtype bf16"
  "int8_g128:--bits 8 --include-embeddings"
  "int6_g128:--bits 6 --include-embeddings"
  "int5_g128:--bits 5 --include-embeddings"
  "int4_g128:--bits 4 --include-embeddings"
  "int3_g128:--bits 3 --include-embeddings"
  "int2_g128:--bits 2 --include-embeddings"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%:*}"; flags="${entry#*:}"
  if [ -f "/workspace/reasonseg/quant/eval/mmr_${tag}.json" ]; then
    echo "[sweep] $tag already done, skipping"; continue
  fi
  echo "[sweep] === $tag ($flags) $(date +%H:%M:%S) ==="
  for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$S $PY "$CODE/run_mmr_quant.py" --tag "$tag" $flags \
      --device cuda:0 --nshards 3 --shard $S $LIM \
      --out "$OUT/mmr_${tag}_s$S.json" > "$OUT/mmr_${tag}_s$S.log" 2>&1 &
  done
  wait
  n=$(ls "$OUT"/mmr_${tag}_s*.json 2>/dev/null | wc -l)
  if [ "$n" -ne 3 ]; then
    echo "[sweep] FAILED $tag: only $n/3 shards"; tail -15 "$OUT"/mmr_${tag}_s*.log; continue
  fi
  $PY "$CODE/merge_quant.py" "mmr_${tag}" "$OUT/mmr_${tag}_s*.json" | grep -E "gIoU|label|merging|wrote"
done
echo "[sweep] all done $(date +%H:%M:%S)"
