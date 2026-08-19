#!/bin/bash
# Phase 3: validate the artifacts that actually ship, and one size optimization.
#
#  from_ckpt   -- evaluate the packed file on disk. Must land on EXACTLY the
#                 int4_g128 row; if it does not, the artifact is not the model
#                 that was measured.
#  bf16rest    -- the 1.8 % of params left unquantized are 69.5 MB of fp32 in a
#                 502 MB file (13.8 %). Rounding them to bf16 halves that; this
#                 measures what it costs, so the smaller file can be offered
#                 honestly rather than assumed free.
#  determinism -- re-run one fp32 shard and diff per-question masks, to establish
#                 that a 0.04 gIoU gap in the tables is a real difference.
set -u
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/quant/code
OUT=/mnt/data0/ameen/quant_out
CK=/mnt/data0/ameen/quant_ckpts
EVAL=/workspace/reasonseg/quant/eval

CONFIGS=(
  # 5.02 bits/weight, against int4_g64's 4.38 -- the clean "spend the budget on
  # codes or on scales?" test, since these two bracket each other in storage
  "int5_pc:--bits 5 --group-size 0 --include-embeddings"
  "int4_g128_bf16rest:--bits 4 --include-embeddings --weight-dtype bf16"
  "int5_g128_bf16rest:--bits 5 --include-embeddings --weight-dtype bf16"
  "int4_from_ckpt:--quant-ckpt $CK/mmr_sam3_int4_g128.pt --quant-mode dense"
)
for entry in "${CONFIGS[@]}"; do
  tag="${entry%%:*}"; flags="${entry#*:}"
  [ -f "$EVAL/mmr_${tag}.json" ] && { echo "[p3] $tag done"; continue; }
  echo "[p3] === $tag ($flags) $(date +%H:%M:%S) ==="
  for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$S $PY "$CODE/run_mmr_quant.py" --tag "$tag" $flags \
      --device cuda:0 --nshards 3 --shard $S \
      --out "$OUT/mmr_${tag}_s$S.json" > "$OUT/mmr_${tag}_s$S.log" 2>&1 &
  done
  wait
  n=$(ls "$OUT"/mmr_${tag}_s*.json 2>/dev/null | wc -l)
  [ "$n" -ne 3 ] && { echo "[p3] FAILED $tag: $n/3 shards"; tail -15 "$OUT"/mmr_${tag}_s*.log; continue; }
  $PY "$CODE/merge_quant.py" "mmr_${tag}" "$OUT/mmr_${tag}_s*.json" | grep -E "gIoU|wrote"
done

echo "[p3] === determinism: re-run fp32 shard 0 ==="
CUDA_VISIBLE_DEVICES=0 $PY "$CODE/run_mmr_quant.py" --tag fp32_rerun --device cuda:0 \
  --nshards 3 --shard 0 --out "$OUT/mmr_fp32rerun_s0.json" > "$OUT/mmr_fp32rerun_s0.log" 2>&1
$PY "$CODE/check_determinism.py" "$OUT/mmr_fp32_s0.json" "$OUT/mmr_fp32rerun_s0.json"
echo "[p3] all done $(date +%H:%M:%S)"
