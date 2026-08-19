#!/usr/bin/env bash
# One GPU worker that pulls the next unclaimed SA-Co/Gold subset for a model and runs it.
# Several workers can share a model safely: claims are atomic mkdir on a lock dir.
#   gold_worker.sh <tag> <gpu-index> [checkpoint]
set -u
TAG=$1; GPU=$2; CKPT=${3:-}
PY=/workspace/envs/sam3/bin/python
OUT=/mnt/data0/ameen/gold_eval/$TAG
LOCKS=$OUT/.claims
mkdir -p "$LOCKS"

# cheapest subsets first, so a worker that gets cut short still maximises subsets covered
SUBSETS="attributes fg_sports_equipment sa1b fg_food crowded metaclip wiki_common"

for S in $SUBSETS; do
  [ -f "$OUT/${S}_metrics.json" ] && continue
  mkdir "$LOCKS/$S" 2>/dev/null || continue     # atomic claim; someone else has it
  echo "=== [$TAG gpu$GPU] $S  $(date +%H:%M:%S)"
  ARGS=(--subset "$S" --device "cuda:0" --out "$OUT/$S.json")
  [ -n "$CKPT" ] && ARGS+=(--checkpoint "$CKPT")
  CUDA_VISIBLE_DEVICES=$GPU $PY /workspace/reasonseg/gold_eval.py "${ARGS[@]}" \
    > "$OUT/$S.log" 2>&1
  if [ -f "$OUT/${S}_metrics.json" ]; then
    rm -f "$OUT/$S.json"
    echo "[$TAG gpu$GPU] $S done $(date +%H:%M:%S)"
  else
    echo "[$TAG gpu$GPU] $S FAILED"; tail -5 "$OUT/$S.log"
    rmdir "$LOCKS/$S" 2>/dev/null                # release so another worker can retry
  fi
done
echo "=== [$TAG gpu$GPU] no unclaimed subsets left $(date +%H:%M:%S)"
