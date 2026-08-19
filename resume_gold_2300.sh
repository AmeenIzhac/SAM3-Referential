#!/usr/bin/env bash
# Sleep until the next 23:00 local, then run the sharded mmr gold subsets.
# atd is not running on this host, so this is the timer. Launch detached:
#   setsid nohup bash resume_gold_2300.sh > <log> 2>&1 < /dev/null &
set -u
CKPT=/mnt/data0/ameen/mmr_runs/mmr_scale/checkpoints/checkpoint.pt
NOW=$(date +%s)
TARGET=$(date -d "today 23:00" +%s)
[ "$TARGET" -le "$NOW" ] && TARGET=$(date -d "tomorrow 23:00" +%s)
echo "now $(date), sleeping $(( (TARGET-NOW)/60 )) min until $(date -d @$TARGET)"
sleep $((TARGET - NOW))
echo "waking $(date)"
exec bash /workspace/reasonseg/run_gold_sharded.sh mmr "$CKPT"
