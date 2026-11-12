"""Step 1 of generate_events.sh: decode <dataset>/yuv/<seq>.yuv to RGB PNGs with the
same conversion as the evaluation (utils/yuv_io.py: BT.709, full range, bilinear
chroma upsampling) and write the input tree of interpolate.py:

    <work>/real/<seq>/0000/images/000000.png ...
    <work>/real/<seq>/0000/source_meta.json

The interpolation factor is ceil(200 / fps), so the simulator runs at >= 200 fps.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from utils.yuv_io import RGB_EVAL_PROFILE, read_yuv_frames  # noqa: E402
from prepare_yuv import canonical_name  # noqa: E402

TARGET_FPS = 200.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yuv-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, default=96)
    ap.add_argument("--sequences", nargs="*", default=None)
    args = ap.parse_args()
    for yuv in sorted(args.yuv_dir.glob("*.yuv")):
        if args.sequences and yuv.stem not in args.sequences:
            continue
        seq, w, h = canonical_name(yuv.stem)
        fps = float(seq.rsplit("_", 1)[1])
        frames = read_yuv_frames(str(yuv), w, h, args.num_frames)
        clip = args.work / "real" / seq / "0000"
        (clip / "images").mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            arr = np.rint(frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(clip / "images" / f"{i:06d}.png")
        sf = max(2, math.ceil(TARGET_FPS / fps))
        meta = {
            "rel_path": f"{seq}/0000", "source_video": seq, "scene_id": "0000",
            "rgb_profile": RGB_EVAL_PROFILE, "width": w, "height": h, "source_fps": fps,
            "source_frames_reported": len(frames), "source_frames": len(frames),
            "sf": sf, "interp_fps": fps * sf, "target_fps": TARGET_FPS,
            "image_pattern": "images/{:06d}.png",
            "interpolated_frames": (len(frames) - 1) * sf + 1,
        }
        (clip / "source_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{seq}: {len(frames)} frames {w}x{h} fps={fps} sf={sf} -> {clip}", flush=True)


if __name__ == "__main__":
    main()
