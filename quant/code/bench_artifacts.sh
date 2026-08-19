#!/bin/bash
# Latency + resident VRAM at each precision, and the packed-vs-dense runtime cost.
# Serial, single GPU, so nothing contends with the timed region.
set -u
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/quant/code
CK=/mnt/data0/ameen/quant_ckpts
OUT=/workspace/reasonseg/quant/eval
mkdir -p "$OUT"

run() {  # run <tag> <flags...>
  tag=$1; shift
  [ -f "$OUT/lat_${tag}.json" ] && { echo "[lat] $tag done"; return; }
  echo "[lat] === $tag ==="
  CUDA_VISIBLE_DEVICES=0 $PY "$CODE/bench_quant_latency.py" --tag "$tag" "$@" \
    --device cuda:0 --out "$OUT/lat_${tag}.json" 2>&1 \
    | grep -E "weights_MB|peak_MB|cold_1img_1q|amortized|median_ms|preloaded"
}

run fp32
run int8_dense --bits 8 --include-embeddings
run int4_dense --bits 4 --include-embeddings
# from the artifact on disk, both runtimes -- dense unpacks once at load, packed
# keeps uint8 codes resident and dequantizes inside every forward
[ -f "$CK/mmr_sam3_int4_g128.pt" ] && {
  run int4_from_ckpt_dense  --quant-ckpt "$CK/mmr_sam3_int4_g128.pt" --quant-mode dense
  # packed keeps the uint8 codes resident and unpacks inside EVERY forward, in
  # plain PyTorch (a bit-plane transpose per weight, no fused kernel). It is
  # slow by construction, so it gets a smaller sample -- the point of the row is
  # the VRAM number and the size of the speed penalty, not a tight median.
  run int4_from_ckpt_packed --quant-ckpt "$CK/mmr_sam3_int4_g128.pt" --quant-mode packed \
      --n-images 6 --n-prompts 12 --warmup 2
}
echo "[lat] all done"
