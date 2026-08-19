#!/usr/bin/env bash
# Run one model over the two large SA-Co/Gold subsets, sharding each across all 3 GPUs.
# Used for the mmr checkpoint, which emits ~90 masks/pair and is ~5x slower than the others.
#   run_gold_sharded.sh <tag> <checkpoint>
set -u
TAG=$1; CKPT=${2:-}
PY=/workspace/envs/sam3/bin/python
OUT=/mnt/data0/ameen/gold_eval/$TAG
N=3
mkdir -p "$OUT"

for S in metaclip wiki_common; do
  if [ -f "$OUT/${S}_metrics.json" ]; then
    echo "[$TAG] $S already scored, skipping"; continue
  fi
  echo "=== [$TAG] $S  $N-way shard  $(date +%H:%M:%S)"
  PIDS=""
  for i in $(seq 0 $((N-1))); do
    ARGS=(--subset "$S" --device cuda:0 --out "$OUT/$S.json" --num-shards $N --shard "$i"
          --conf 0.45)   # provably identical metrics to 0.05, ~25x less mask encoding
    [ -n "$CKPT" ] && ARGS+=(--checkpoint "$CKPT")
    CUDA_VISIBLE_DEVICES=$i $PY /workspace/reasonseg/gold_eval.py "${ARGS[@]}" \
      > "$OUT/$S.shard$i.log" 2>&1 &
    PIDS="$PIDS $!"
  done
  FAIL=0
  for p in $PIDS; do wait "$p" || FAIL=1; done
  if [ "$FAIL" != 0 ]; then
    echo "[$TAG] $S: a shard failed, not merging"; tail -5 "$OUT/$S".shard*.log; continue
  fi
  echo "=== [$TAG] $S merging + scoring  $(date +%H:%M:%S)"
  $PY /workspace/reasonseg/gold_eval.py --subset "$S" --out "$OUT/$S.json" \
    --merge-shards $N > "$OUT/$S.log" 2>&1
  if [ -f "$OUT/${S}_metrics.json" ]; then
    rm -f "$OUT/$S.json" "$OUT/$S".json.shard*      # raw predictions are ~7-14 GB here
    echo "[$TAG] $S done $(date +%H:%M:%S)"
  else
    echo "[$TAG] $S SCORING FAILED"; tail -20 "$OUT/$S.log"
  fi
done
echo "=== [$TAG] sharded subsets finished $(date +%H:%M:%S)"
