#!/bin/bash
# Phase 4: mixed precision, the payoff of the subsystem probe.
#
# §6 found the detection heads carry 21x the per-parameter precision sensitivity
# of the vision backbone while being only 3.3 % of the quantized parameters. So
# holding just those at 8 bits costs almost nothing in storage:
#   int3 body + int8 heads = (798.0*3.19 + 27.5*8.19)/825.5 = 3.36 bits/weight
#   int4 body + int8 heads = (798.0*4.19 + 27.5*8.19)/825.5 = 4.32 bits/weight
# against uniform int3's 3.19 and int4's 4.19. If the sensitivity result is real,
# these should beat the uniform rows at nearly the same size.
set -u
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/quant/code
OUT=/mnt/data0/ameen/quant_out
EVAL=/workspace/reasonseg/quant/eval

CONFIGS=(
  "int4_g128_head8:--bits 4 --head-bits 8 --include-embeddings"
  "int3_g128_head8:--bits 3 --head-bits 8 --include-embeddings"
  "int3_g64_head8:--bits 3 --group-size 64 --head-bits 8 --include-embeddings"
)
for entry in "${CONFIGS[@]}"; do
  tag="${entry%%:*}"; flags="${entry#*:}"
  [ -f "$EVAL/mmr_${tag}.json" ] && { echo "[p4] $tag done"; continue; }
  echo "[p4] === $tag ($flags) $(date +%H:%M:%S) ==="
  for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$S $PY "$CODE/run_mmr_quant.py" --tag "$tag" $flags \
      --device cuda:0 --nshards 3 --shard $S \
      --out "$OUT/mmr_${tag}_s$S.json" > "$OUT/mmr_${tag}_s$S.log" 2>&1 &
  done
  wait
  n=$(ls "$OUT"/mmr_${tag}_s*.json 2>/dev/null | wc -l)
  [ "$n" -ne 3 ] && { echo "[p4] FAILED $tag: $n/3 shards"; tail -15 "$OUT"/mmr_${tag}_s*.log; continue; }
  $PY "$CODE/merge_quant.py" "mmr_${tag}" "$OUT/mmr_${tag}_s*.json" | grep -E "gIoU|wrote"
done
echo "[p4] all done $(date +%H:%M:%S)"
