#!/usr/bin/env bash
# One model, all SA-Co/Gold subsets, on one GPU.
#   run_gold.sh <tag> <gpu-index> [checkpoint]      (omit checkpoint => stock facebook/sam3)
set -u
TAG=$1; GPU=$2; CKPT=${3:-}
PY=/workspace/envs/sam3/bin/python
OUT=/mnt/data0/ameen/gold_eval/$TAG
mkdir -p "$OUT"

# sa1b first: it is the subset with an existing baseline to cross-check the harness against
SUBSETS="sa1b metaclip attributes crowded fg_food fg_sports_equipment wiki_common"

for S in $SUBSETS; do
  if [ -f "$OUT/${S}_metrics.json" ]; then
    echo "[$TAG] $S already scored, skipping"; continue
  fi
  echo "=== [$TAG] $S  $(date +%H:%M:%S)"
  ARGS=(--subset "$S" --device "cuda:0" --out "$OUT/$S.json")
  [ -n "$CKPT" ] && ARGS+=(--checkpoint "$CKPT")
  CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/mnt/data0/ameen/hf_cache \
    $PY /workspace/reasonseg/gold_eval.py "${ARGS[@]}" > "$OUT/$S.log" 2>&1
  if [ -f "$OUT/${S}_metrics.json" ]; then
    rm -f "$OUT/$S.json"          # raw predictions are large; metrics are the deliverable
    echo "[$TAG] $S done"
  else
    echo "[$TAG] $S FAILED, see $OUT/$S.log"; tail -5 "$OUT/$S.log"
  fi
done
echo "=== [$TAG] all subsets finished $(date +%H:%M:%S)"
