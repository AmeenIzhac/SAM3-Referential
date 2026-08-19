#!/bin/bash
# Benchmark the rolling training checkpoint on MMR val (question prompt), 3-way
# sharded across the (paused) training GPUs, then merge into one summary json.
#
# Called by the SAM3 trainer's inline-bench hook as:  bench_scale.sh <ckpt> <tag>
# Training is paused at a barrier while this runs, and every rank has already
# called torch.cuda.empty_cache(), so ~13 GB is free on each GPU.
#
# Env: BENCH_LIMIT=<n>  -> only evaluate n val images per shard (smoke tests).
set -u
ckpt=$1; tag=$2
PY=/workspace/envs/sam3/bin/python
CODE=/workspace/reasonseg/mmrcomp/code
OUT=/mnt/data0/ameen/mmr_out/scale
mkdir -p "$OUT"
cd /workspace/reasonseg

lim=""
[ -n "${BENCH_LIMIT:-}" ] && lim="--limit $BENCH_LIMIT"

echo "[bench_scale] tag=$tag ckpt=$ckpt $(date)"
for S in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$S $PY "$CODE/run_sam3_mmr.py" --ckpt-path "$ckpt" \
    --prompt-mode question --device cuda:0 --nshards 3 --shard $S $lim \
    --out "$OUT/mmrscale_${tag}_s$S.json" > "$OUT/mmrscale_${tag}_s$S.log" 2>&1 &
done
wait

n=$(ls "$OUT"/mmrscale_${tag}_s*.json 2>/dev/null | wc -l)
if [ "$n" -ne 3 ]; then
  echo "[bench_scale] FAILED: only $n/3 shards produced output for $tag"
  tail -20 "$OUT"/mmrscale_${tag}_s*.log
  exit 1
fi

$PY "$CODE/merge_mmr.py" "mmrscale_${tag}" "$OUT/mmrscale_${tag}_s*.json"
echo "[bench_scale] $tag merged $(date)"
