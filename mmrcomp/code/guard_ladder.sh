#!/bin/bash
# Watchdog + artifact refresher for the MMR scaling ladder.
#
# Run 1 of this ladder burned 15 h and produced two worthless rungs because the
# trainer silently stopped applying optimizer steps (NaN gradients from the fused
# Triton focal kernel) while still logging a healthy loss. The trainer now aborts
# itself on a sustained skip rate; this script is the outer belt-and-braces:
#
#   * regenerates the table/plot whenever a new rung lands
#   * ALERTs if grad-skips appear at all, and loudly if they are climbing
#   * ALERTs if a rung's weight fingerprint is unchanged (rung is not a data point)
#   * ALERTs if the training log stops advancing (hang) while the process is alive
#   * does a final regen and exits once the trainer is gone
#
# Everything it flags is written to STDOUT and to $ALERTS, so a check of one file
# tells you whether the run is healthy.
set -u
EVAL=/workspace/reasonseg/mmr_eval
OUT=/workspace/reasonseg/mmrcomp
LOG=/mnt/data0/ameen/mmr_runs/mmr_scale_train.log
ALERTS=/mnt/data0/ameen/mmr_runs/ladder_alerts.log
PY=/workspace/envs/sam3/bin/python
STALL_SECS=${STALL_SECS:-1800}      # bench pauses are ~6 min, so 30 min is generous

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$ALERTS"; }
alert() { echo "$(date +%H:%M:%S) *** ALERT: $*" | tee -a "$ALERTS"; }

: > "$ALERTS"
say "guard started; watching $LOG"
last_rungs=-1; last_skips=0; last_step=""; last_step_change=$(date +%s)

while true; do
  # --- refresh artifacts when a rung lands ---
  n=$(ls "$EVAL"/mmrscale_*.json 2>/dev/null | wc -l)
  if [ "$n" -ne "$last_rungs" ] && [ "$n" -gt 0 ]; then
    "$PY" "$OUT/code/plot_ladder.py" > "$OUT/scaling_ladder_table.md" 2>&1 \
      && say "regenerated artifacts with $n rungs" \
      || alert "plot_ladder.py failed with $n rungs"
    last_rungs=$n
  fi

  if [ -f "$LOG" ]; then
    # --- gradient skips: none are expected now; any are worth surfacing ---
    # grep -c prints 0 and exits 1 when there are no matches, so a `|| echo 0`
    # here would append a SECOND 0 and break every later integer test.
    skips=$(grep -c "skipping optimizer step" "$LOG" 2>/dev/null); skips=${skips:-0}
    if [ "$skips" -gt "$last_skips" ]; then
      alert "grad-skips rose ${last_skips} -> ${skips} (NaN gradients are back)"
      grep "grad-diag" "$LOG" 2>/dev/null | tail -4 | tee -a "$ALERTS"
      last_skips=$skips
    fi

    # --- a rung whose weights did not move is not a data point ---
    if grep -q "WEIGHTS UNCHANGED" "$LOG" 2>/dev/null; then
      alert "trainer reported WEIGHTS UNCHANGED at a rung -- run is not learning"
      grep "WEIGHTS UNCHANGED" "$LOG" | tail -2 | tee -a "$ALERTS"
    fi

    # --- hang detection ---
    step=$(grep -oE "Train Epoch: \[0\]\[ *[0-9]+/" "$LOG" 2>/dev/null | tail -1)
    now=$(date +%s)
    if [ "$step" != "$last_step" ]; then
      last_step="$step"; last_step_change=$now
    elif [ $((now - last_step_change)) -gt "$STALL_SECS" ] \
         && pgrep -f "configs/newsam/mmr_scale.yaml" > /dev/null 2>&1; then
      alert "no progress for $((now - last_step_change))s (last: ${step:-none}) -- possible hang"
      last_step_change=$now
    fi
  fi

  # --- exit once the trainer is gone and the last rung is folded in ---
  if ! pgrep -f "configs/newsam/mmr_scale.yaml" > /dev/null 2>&1; then
    sleep 240
    n2=$(ls "$EVAL"/mmrscale_*.json 2>/dev/null | wc -l)
    if [ "$n2" -eq "$n" ]; then
      "$PY" "$OUT/code/plot_ladder.py" > "$OUT/scaling_ladder_table.md" 2>&1
      say "trainer gone; final regen with $n2 rungs; total grad-skips ${last_skips}"
      [ "${last_skips:-0}" -eq 0 ] && say "run finished CLEAN (zero skipped optimizer steps)" \
                                   || alert "run finished with ${last_skips} skipped steps"
      exit 0
    fi
  fi
  sleep 60
done
