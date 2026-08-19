#!/bin/bash
# Write the packed k-bit checkpoints and report real on-disk sizes.
# Serial on one GPU: the fit is a few seconds of GPU per tensor, the write is
# CPU/disk bound, and running these in parallel just contends for the disk.
set -u
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/quant/code
CK=/mnt/data0/ameen/quant_ckpts
mkdir -p "$CK"

CONFIGS=(
  "int8_g128:--bits 8 --group-size 128 --include-embeddings"
  "int6_g128:--bits 6 --group-size 128 --include-embeddings"
  "int5_g128:--bits 5 --group-size 128 --include-embeddings"
  "int4_g128:--bits 4 --group-size 128 --include-embeddings"
  "int4_g64:--bits 4 --group-size 64 --include-embeddings"
  "int3_g128:--bits 3 --group-size 128 --include-embeddings"
  # int2 is not a usable model (0.35 gIoU); written only so the packed-mode
  # latency curve in quant.md §9 has a 2-bit point.
  "int2_g128:--bits 2 --group-size 128 --include-embeddings"
)
for entry in "${CONFIGS[@]}"; do
  tag="${entry%%:*}"; flags="${entry#*:}"
  out="$CK/mmr_sam3_${tag}.pt"
  [ -f "$out" ] && { echo "[artifact] $tag exists ($(du -h "$out" | cut -f1))"; continue; }
  echo "[artifact] === $tag ==="
  CUDA_VISIBLE_DEVICES=0 $PY "$CODE/quantize_ckpt.py" $flags --out "$out" \
    2>&1 | grep -v -i "warning\|pkg_resources"
done
echo "[artifact] sizes:"; ls -la "$CK"
echo "[artifact] all done $(date +%H:%M:%S)"
