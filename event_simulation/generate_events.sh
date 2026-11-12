#!/usr/bin/env bash
# Synthetic events for one test set, as used in the paper:
#   <dataset>/yuv/<seq>.yuv --make_frames.py--> RGB PNGs (BT.709, as in evaluation)
#     --interpolate.py (Super-SloMo, ceil(200/fps)x, reflect-pad to a multiple of 32)-->
#     --simulate_events.py (DVS-Voltmeter, DVS346 parameters)--> one .npz per frame interval
# Output: <dataset>/events/<seq>/000000.npz ... (file k = events between frames k and k+1)
#
#   CKPT=/path/to/SuperSloMo.ckpt DATASET_DIR=/path/to/ENVC_testsets/HEVC-D bash event_simulation/generate_events.sh
#
# Env: CKPT, DATASET_DIR (required); NF (frames, default 96); PY (default python);
#      WORK (scratch dir, default $DATASET_DIR/_work, removed at the end unless KEEP_WORK=1).
# Super-SloMo runs on the GPU when one is visible. The simulator is stochastic, so a
# rerun gives statistically equivalent but not bit-identical events; use the released
# events to reproduce the reported numbers exactly.
set -euo pipefail
: "${CKPT:?set CKPT=/path/to/SuperSloMo.ckpt}"
: "${DATASET_DIR:?set DATASET_DIR=/path/to/<dataset> (contains yuv/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
NF=${NF:-96}
W=${WORK:-$DATASET_DIR/_work}
OUT=$DATASET_DIR/events

"$PY" "$HERE/make_frames.py" --yuv-dir "$DATASET_DIR/yuv" --work "$W" --num-frames "$NF"
"$PY" "$HERE/interpolate.py" --input_root "$W/real" --output_root "$W/interp" \
  --checkpoint "$CKPT" --sf 0 --target_fps 200 --pad32 --skip_existing
"$PY" "$HERE/simulate_events.py" --input_root "$W/interp" --output_root "$W/real" --skip_done

for clip in "$W"/real/*/0000; do
  s=$(basename "$(dirname "$clip")")
  n_img=$(ls "$clip/images" | wc -l); n_ev=$(ls "$clip/events" | wc -l)
  if [ "$n_ev" -ne $(( n_img - 1 )) ]; then echo "[error] $s: $n_img frames but $n_ev event files" >&2; exit 1; fi
  rm -rf "${OUT:?}/${s:?}"; mkdir -p "$OUT/$s"
  cp "$clip"/events/*.npz "$OUT/$s/"
  echo "[done] $s: $n_ev event files -> $OUT/$s"
done
if [ -z "${KEEP_WORK:-}" ]; then rm -rf "${W:?}"; fi
