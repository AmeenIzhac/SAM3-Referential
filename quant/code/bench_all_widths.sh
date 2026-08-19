#!/bin/bash
# Forward-pass speed at every width, in both runtimes.
#
# dense is expected to be identical everywhere -- it unpacks to ordinary fp32
# weights at load, so the compute graph does not know it was quantized. Measured
# rather than asserted, because "should be identical" is how you miss things.
#
# packed is NOT expected to be identical: unpack_bits runs one shift-and-or pass
# per bit, inside every forward, so its cost should scale with the bit width.
set -u
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/quant/code
CK=/mnt/data0/ameen/quant_ckpts
OUT=/workspace/reasonseg/quant/eval

run() { tag=$1; shift
  [ -f "$OUT/lat_${tag}.json" ] && { echo "[lat] $tag done"; return; }
  echo "[lat] === $tag ==="
  CUDA_VISIBLE_DEVICES=0 $PY "$CODE/bench_quant_latency.py" --tag "$tag" "$@" \
    --device cuda:0 --out "$OUT/lat_${tag}.json" 2>&1 | grep -E "mean_ms|weights_MB|amortized"
}

# --- dense, on the fly, every width ---
run int6_dense --bits 6 --include-embeddings
run int5_dense --bits 5 --include-embeddings
run int3_dense --bits 3 --include-embeddings
run int2_dense --bits 2 --include-embeddings
run int4_g32_dense --bits 4 --group-size 32 --include-embeddings

# --- packed, from the artifacts; fewer samples because it is slow by design ---
for b in 8 6 5 4 3; do
  f="$CK/mmr_sam3_int${b}_g128.pt"
  [ -f "$f" ] && run "int${b}_packed" --quant-ckpt "$f" --quant-mode packed \
      --n-images 6 --n-prompts 12 --warmup 2
done
echo "[lat] all widths done $(date +%H:%M:%S)"
