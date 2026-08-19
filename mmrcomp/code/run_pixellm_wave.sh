#!/bin/bash
# Full PixelLM-7B evaluation on MMR val, 3-way sharded, free generation only.
# (Teacher forcing is deliberately not run: PixelLM uses a 6-token codebook per
# target, which MMR's one-`{seg}`-per-target answer text cannot address. See the
# header of run_pixellm_mmr.py.)
#
# This box is shared. The first attempt at this run lost ~5 h when another job
# appeared and took 9.5 GB/GPU while PixelLM needed ~14 GB. Each shard therefore
# flushes partial results and is retried with --resume, so contention costs
# minutes rather than the whole run.
set -u
PY=/mnt/data0/ameen/envs/m2sa/bin/python
CODE=/workspace/reasonseg/mmrcomp/code
PIXELLM=/tmp/claude-1009/-workspace-reasonseg/d6cd4b24-35e9-4fa3-8cad-354194dce19a/scratchpad/PixelLM
OUT=/mnt/data0/ameen/mmr_out
MAX_TRIES=${MAX_TRIES:-8}
mkdir -p "$OUT"
cd "$PIXELLM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_shard() {
  local S=$1
  for try in $(seq 1 "$MAX_TRIES"); do
    CUDA_VISIBLE_DEVICES=$S $PY "$CODE/run_pixellm_mmr.py" \
      --device cuda:0 --nshards 3 --shard "$S" --resume \
      --out "$OUT/pixellm_gen_s$S.json" >> "$OUT/pixellm_gen_s$S.log" 2>&1
    if $PY - "$OUT/pixellm_gen_s$S.json" <<'EOF'
import json,sys
try: sys.exit(0 if json.load(open(sys.argv[1]))["summary"].get("complete") else 1)
except Exception: sys.exit(1)
EOF
    then
      echo "shard $S complete (try $try)"; return 0
    fi
    echo "shard $S incomplete after try $try; retrying in 120s"
    sleep 120
  done
  echo "!!! shard $S never completed after $MAX_TRIES tries"; return 1
}

echo "===== PixelLM gen  $(date) ====="
for S in 0 1 2; do run_shard "$S" & done
wait

for S in 0 1 2; do
  $PY - "$OUT/pixellm_gen_s$S.json" <<'EOF' || { echo "!!! shard missing/incomplete"; exit 1; }
import json,sys
sys.exit(0 if json.load(open(sys.argv[1]))["summary"].get("complete") else 1)
EOF
done
$PY "$CODE/merge_mmr.py" "pixellm_gen" "$OUT/pixellm_gen_s*.json"
echo "===== PixelLM DONE  $(date) ====="
