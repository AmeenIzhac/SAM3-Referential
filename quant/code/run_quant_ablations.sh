#!/bin/bash
# Phase 2: what recovers a lossy width, and which subsystem is precision-critical.
#
#  * group size    -- per-channel (the granularity every stock int8 path uses)
#                     -> g128 -> g64 -> g32: more scales, finer fit, bigger
#                     artifact. Run at 8 bits (where nothing should matter) and
#                     at 4 (where it should).
#  * embeddings    -- is the 49408x1024 CLIP token table safe to quantize?
#  * subsystem     -- quantize ONLY the vision backbone / ONLY the language
#                     backbone / ONLY the detection heads at a punishing width.
#                     mmr.md's thesis is that SAM3's MMR gain is a language
#                     effect, not a pixel effect; if that is right, the language
#                     side should be the precision-critical one per parameter.
set -u
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/quant/code
OUT=/mnt/data0/ameen/quant_out
mkdir -p "$OUT"
BITS="${PROBE_BITS:-3}"

CONFIGS=(
  "int8_pc:--bits 8 --group-size 0 --include-embeddings"
  "int4_pc:--bits 4 --group-size 0 --include-embeddings"
  "int4_g64:--bits 4 --group-size 64 --include-embeddings"
  "int4_g32:--bits 4 --group-size 32 --include-embeddings"
  "int4_g128_noemb:--bits 4"
  "int3_g64:--bits 3 --group-size 64 --include-embeddings"
  "int3_g32:--bits 3 --group-size 32 --include-embeddings"
  "int${BITS}_vision_only:--bits ${BITS} --only vision_backbone"
  "int${BITS}_language_only:--bits ${BITS} --include-embeddings --only language_backbone"
  "int${BITS}_heads_only:--bits ${BITS} --only ^transformer ^geometry_encoder ^segmentation_head ^dot_prod_scoring"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%:*}"; flags="${entry#*:}"
  if [ -f "/workspace/reasonseg/quant/eval/mmr_${tag}.json" ]; then
    echo "[abl] $tag already done, skipping"; continue
  fi
  echo "[abl] === $tag ($flags) $(date +%H:%M:%S) ==="
  for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$S $PY "$CODE/run_mmr_quant.py" --tag "$tag" $flags \
      --device cuda:0 --nshards 3 --shard $S \
      --out "$OUT/mmr_${tag}_s$S.json" > "$OUT/mmr_${tag}_s$S.log" 2>&1 &
  done
  wait
  n=$(ls "$OUT"/mmr_${tag}_s*.json 2>/dev/null | wc -l)
  if [ "$n" -ne 3 ]; then
    echo "[abl] FAILED $tag: only $n/3 shards"; tail -15 "$OUT"/mmr_${tag}_s*.log; continue
  fi
  $PY "$CODE/merge_quant.py" "mmr_${tag}" "$OUT/mmr_${tag}_s*.json" | grep -E "gIoU|label|merging|wrote"
done
echo "[abl] all done $(date +%H:%M:%S)"
