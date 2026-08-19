#!/bin/bash
# Regenerate the ladder table + plot whenever a new rung lands, so
# mmrcomp/{scaling_curve_full.png,scaling_ladder.json,scaling_ladder_table.md}
# are always current while the ~14 h run is in flight.
set -u
EVAL=/workspace/reasonseg/mmr_eval
OUT=/workspace/reasonseg/mmrcomp
PY=/workspace/envs/sam3/bin/python
last=-1
while true; do
  n=$(ls "$EVAL"/mmrscale_*.json 2>/dev/null | wc -l)
  if [ "$n" -ne "$last" ] && [ "$n" -gt 0 ]; then
    "$PY" "$OUT/code/plot_ladder.py" > "$OUT/scaling_ladder_table.md" 2>&1 \
      && echo "$(date +%H:%M) regenerated with $n rungs" \
      || echo "$(date +%H:%M) plot failed with $n rungs"
    last=$n
  fi
  # stop once the trainer is gone and the last rung has been folded in
  if ! pgrep -f "configs/newsam/mmr_scale.yaml" > /dev/null 2>&1; then
    sleep 240
    n2=$(ls "$EVAL"/mmrscale_*.json 2>/dev/null | wc -l)
    if [ "$n2" -eq "$n" ]; then
      "$PY" "$OUT/code/plot_ladder.py" > "$OUT/scaling_ladder_table.md" 2>&1
      echo "$(date +%H:%M) trainer gone; final regen with $n2 rungs; exiting"
      exit 0
    fi
  fi
  sleep 60
done
