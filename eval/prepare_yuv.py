"""Cut the first N frames of the official 8-bit 4:2:0 test sequences and store them
under the names the evaluation and the released events use:

    python eval/prepare_yuv.py --src /path/to/official/UVG --out /path/to/ENVC_testsets/UVG/yuv

    Beauty_1920x1080_120fps_420_8bit_YUV.yuv -> Beauty_1920x1080_120.yuv
    BQTerrace_1920x1080_60.yuv               -> BQTerrace_1920x1080_60.yuv
    videoSRC01_1920x1080_30.yuv              -> videoSRC01_1920x1080_30.yuv

The name is <prefix>_<W>x<H>_<fps>; frame size and frame rate are read from it.
"""
import argparse
import re
from pathlib import Path


def canonical_name(stem: str):
    m = re.search(r"_(\d+)x(\d+)", stem)
    if not m:
        raise ValueError(f"cannot find _<W>x<H> in {stem}")
    f = re.match(r"_(\d+(?:\.\d+)?)(?:fps|Hz)?(?:_|$)", stem[m.end():])
    if not f:
        raise ValueError(f"cannot find the frame rate after _<W>x<H> in {stem}")
    fps = float(f[1])
    fps_txt = str(int(fps)) if fps.is_integer() else f[1]
    return f"{stem[:m.start()]}_{m[1]}x{m[2]}_{fps_txt}", int(m[1]), int(m[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder with the official .yuv files")
    ap.add_argument("--out", type=Path, required=True, help="<dataset>/yuv")
    ap.add_argument("--num-frames", type=int, default=96)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.yuv"))
    if not files:
        raise SystemExit(f"no .yuv files in {args.src}")
    for path in files:
        name, w, h = canonical_name(path.stem)
        frame_bytes = w * h * 3 // 2
        size = path.stat().st_size
        if size % frame_bytes:
            raise SystemExit(f"{path.name}: {size} bytes is not a whole number of 8-bit 4:2:0 {w}x{h} frames")
        if size // frame_bytes < args.num_frames:
            raise SystemExit(f"{path.name}: only {size // frame_bytes} frames, need {args.num_frames}")
        with path.open("rb") as src, (args.out / f"{name}.yuv").open("wb") as dst:
            dst.write(src.read(frame_bytes * args.num_frames))
        print(f"{path.name} -> {name}.yuv ({args.num_frames} frames, {w}x{h})")


if __name__ == "__main__":
    main()
