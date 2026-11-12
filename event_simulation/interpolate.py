#!/usr/bin/env python3
"""
Interpolate native-rate PNG sequences with Super-SloMo (step 2 of generate_events.sh).

Input layout:
  <input_root>/{group}/{video_id}/images/000000.png ...
  <input_root>/{group}/{video_id}/source_meta.json

Output layout for DVS-Voltmeter:
  <output_root>/{group}/{video_id}/images/000000.png ... high-fps timeline
  <output_root>/{group}/{video_id}/images/info.txt
  <output_root>/{group}/{video_id}/source_meta.json

This script intentionally does not depend on cv2 or tqdm, because the legacy
super-slomo conda env often only has PIL/numpy/torch/torchvision.
"""
import argparse
import inspect
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "third_party", "superslomo"))

import model  # noqa: E402

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = getattr(Image, "LANCZOS", Image.ANTIALIAS)


def parse_args():
    p = argparse.ArgumentParser(description="Batch Super-SloMo interpolation of PNG sequences.")
    p.add_argument(
        "--input_root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output_root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    p.add_argument("--sf", type=int, default=0, help="Fixed slomo factor. 0 uses source_meta sf/target_fps.")
    p.add_argument("--target_fps", type=float, default=200.0)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest with rel_path rows. start_index/limit apply to manifest rows.",
    )
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_cudnn", action="store_true", help="Disable cuDNN conv kernels.")
    p.add_argument(
        "--pad32",
        action="store_true",
        help="Reflect-pad frames up to a multiple of 32 and crop back, instead of "
        "Lanczos-resizing down to a multiple of 32 and bilinear-resizing back. The "
        "resize round trip blurs interpolated frames but not the copied native "
        "frames, which shows up as spurious events at low resolutions (e.g. 240->224 "
        "for HEVC-D). Off by default so the training pipeline is unchanged.",
    )
    p.add_argument(
        "--no_cudnn_benchmark",
        action="store_true",
        help="Set cudnn.benchmark=False. Default uses benchmark=True for fixed-size clips.",
    )
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


def calc_sf(src_fps, fixed_sf, target_fps, meta_sf=0):
    if fixed_sf and fixed_sf >= 2:
        return int(fixed_sf)
    if meta_sf and int(meta_sf) >= 2:
        return int(meta_sf)
    if src_fps and src_fps > 0:
        return max(2, int(math.ceil(float(target_fps) / float(src_fps))))
    return 9


def load_checkpoint(path):
    kwargs = {"map_location": "cpu"}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
    except (AttributeError, TypeError, ValueError):
        pass
    return torch.load(str(path), **kwargs)


def build_transforms(device):
    mean = [0.429, 0.431, 0.397]
    std = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean, std=std)
    negmean = [x * -1 for x in mean]
    rev_normalize = transforms.Normalize(mean=negmean, std=std)
    if device.type == "cpu":
        transform = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        tp = transforms.Compose([rev_normalize, transforms.ToPILImage()])
    return transform, tp


def configure_cudnn(args, device):
    if device.type != "cuda":
        return
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
        print("[model] cudnn disabled (--no_cudnn)", flush=True)
        return
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
    try:
        torch.backends.cudnn.deterministic = bool(args.no_cudnn_benchmark)
    except (AttributeError, RuntimeError):
        pass
    print(
        "[model] cudnn enabled benchmark={}".format(torch.backends.cudnn.benchmark),
        flush=True,
    )


def read_manifest(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if "rel_path" not in row:
                raise ValueError("{}:{} missing rel_path".format(path, line_no))
            rows.append(row)
    return rows


def collect_manifest_clips(input_root, manifest, start_index, limit):
    rows = read_manifest(manifest)
    if start_index > 0:
        rows = rows[start_index:]
    if limit and limit > 0:
        rows = rows[:limit]

    clips = []
    for row in rows:
        clip = input_root / row["rel_path"]
        if not (clip / "images").is_dir() or not (clip / "source_meta.json").is_file():
            raise FileNotFoundError("manifest clip is not extracted yet: {}".format(clip))
        clips.append(clip)
    return clips


def collect_clips(input_root, start_index, limit):
    clips = []
    for first in sorted(input_root.iterdir()):
        if not first.is_dir():
            continue
        for second in sorted(first.iterdir()):
            if (second / "images").is_dir() and (second / "source_meta.json").is_file():
                clips.append(second)
    if start_index > 0:
        clips = clips[start_index:]
    if limit and limit > 0:
        clips = clips[:limit]
    return clips


def expected_interp_frames(source_frames, sf):
    if source_frames < 1:
        return 0
    return (source_frames - 1) * sf + 1


def output_complete(out_images_dir, expected_frames):
    if expected_frames <= 0 or not out_images_dir.is_dir():
        return False
    if not (out_images_dir / "info.txt").is_file():
        return False
    return len(list(out_images_dir.glob("*.png"))) >= expected_frames


def output_frame_path(out_images_dir, frame_index):
    return out_images_dir / "{:06d}.png".format(frame_index)


def consecutive_png_count(out_images_dir):
    count = 0
    while output_frame_path(out_images_dir, count).is_file():
        count += 1
    return count


def remove_pngs_from(out_images_dir, start_index):
    removed = 0
    for path in out_images_dir.glob("*.png"):
        try:
            idx = int(path.stem)
        except ValueError:
            continue
        if idx >= start_index:
            path.unlink()
            removed += 1
    return removed


def to_tensor(path, transform, model_dim, device, pad=False):
    with Image.open(path) as img:
        img = img.convert("RGB")
        orig_dim = img.size
        if pad:
            arr = np.asarray(img)
            pad_h = model_dim[1] - arr.shape[0]
            pad_w = model_dim[0] - arr.shape[1]
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            resized = Image.fromarray(arr)
        else:
            resized = img.resize(model_dim, _LANCZOS)
    return transform(resized).unsqueeze(0).to(device), orig_dim


def interpolate_pair(I0, I1, sf, flow_comp, arb_flow, flow_back_warp):
    flow_out = flow_comp(torch.cat((I0, I1), dim=1))
    f_0_1 = flow_out[:, :2, :, :]
    f_1_0 = flow_out[:, 2:, :, :]
    out = []
    for intermediate_index in range(1, sf):
        t = float(intermediate_index) / sf
        temp = -t * (1 - t)
        f_coeff = [temp, t * t, (1 - t) * (1 - t), temp]

        f_t_0 = f_coeff[0] * f_0_1 + f_coeff[1] * f_1_0
        f_t_1 = f_coeff[2] * f_0_1 + f_coeff[3] * f_1_0

        g_i0_f_t_0 = flow_back_warp(I0, f_t_0)
        g_i1_f_t_1 = flow_back_warp(I1, f_t_1)

        intrp_out = arb_flow(
            torch.cat((I0, I1, f_0_1, f_1_0, f_t_1, f_t_0, g_i1_f_t_1, g_i0_f_t_0), dim=1)
        )

        f_t_0_f = intrp_out[:, :2, :, :] + f_t_0
        f_t_1_f = intrp_out[:, 2:4, :, :] + f_t_1
        v_t_0 = torch.sigmoid(intrp_out[:, 4:5, :, :])
        v_t_1 = 1 - v_t_0

        g_i0_f_t_0_f = flow_back_warp(I0, f_t_0_f)
        g_i1_f_t_1_f = flow_back_warp(I1, f_t_1_f)

        w_coeff = [1 - t, t]
        denom = w_coeff[0] * v_t_0 + w_coeff[1] * v_t_1
        ft_p = (w_coeff[0] * v_t_0 * g_i0_f_t_0_f + w_coeff[1] * v_t_1 * g_i1_f_t_1_f) / denom
        out.append(ft_p)
    return out


def interpolate_pair_with_cudnn_fallback(I0, I1, sf, flow_comp, arb_flow, flow_back_warp, rel, pair_idx):
    try:
        return interpolate_pair(I0, I1, sf, flow_comp, arb_flow, flow_back_warp)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "cudnn" not in msg or not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
            raise
        print(
            "[warn] CuDNN RuntimeError at {} pair {}: {}. Disabling cuDNN and retrying.".format(
                rel, pair_idx, exc
            ),
            flush=True,
        )
        torch.backends.cudnn.enabled = False
        try:
            torch.cuda.empty_cache()
        except (AttributeError, RuntimeError):
            pass
        return interpolate_pair(I0, I1, sf, flow_comp, arb_flow, flow_back_warp)


def save_info_line(handle, path, frame_index, interp_fps):
    ts = int(round(frame_index * 1_000_000.0 / interp_fps))
    handle.write("{} {:012d}\n".format(path.resolve(), ts))


def rewrite_info_txt(out_images_dir, frame_count, interp_fps):
    info_path = out_images_dir / "info.txt"
    with info_path.open("w", encoding="utf-8") as info:
        for frame_index in range(frame_count):
            save_info_line(info, output_frame_path(out_images_dir, frame_index), frame_index, interp_fps)


def save_pil(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def clear_output(out_dir):
    if out_dir.is_dir():
        shutil.rmtree(out_dir)


def prepare_resume(out_images, expected, sf, interp_fps, overwrite):
    if overwrite:
        return 0, 0

    existing = consecutive_png_count(out_images)
    if existing <= 0:
        return 0, 0

    existing = min(existing, expected)
    keep = ((existing - 1) // sf) * sf + 1
    removed = remove_pngs_from(out_images, keep)
    rewrite_info_txt(out_images, keep, interp_fps)

    resume_pairs = (keep - 1) // sf
    if removed > 0:
        print(
            "[resume] trimmed {} partial frame(s); keeping {} complete output frame(s)".format(
                removed, keep
            ),
            flush=True,
        )
    print(
        "[resume] found {} complete source pair(s); continuing at pair {}".format(
            resume_pairs, resume_pairs + 1
        ),
        flush=True,
    )
    return resume_pairs, keep


def process_clip(clip_dir, input_root, output_root, args, nets):
    rel = clip_dir.relative_to(input_root)
    out_clip = output_root / rel
    out_images = out_clip / "images"

    with (clip_dir / "source_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    src_fps = float(meta.get("source_fps") or 0.0)
    sf = calc_sf(src_fps, args.sf, args.target_fps, meta.get("sf", 0))
    src_paths = sorted((clip_dir / "images").glob("*.png"))
    if len(src_paths) < 2:
        return "skip_too_few_frames", rel, 0, sf

    expected = expected_interp_frames(len(src_paths), sf)
    if args.skip_existing and output_complete(out_images, expected):
        return "skip_existing", rel, expected, sf

    if args.overwrite:
        clear_output(out_clip)
    out_images.mkdir(parents=True, exist_ok=True)

    interp_fps = src_fps * sf if src_fps > 0 else float(args.target_fps)
    resume_pairs, out_idx = prepare_resume(out_images, expected, sf, interp_fps, args.overwrite)
    if resume_pairs >= len(src_paths) - 1 and output_complete(out_images, expected):
        return "skip_existing", rel, expected, sf

    device, transform, tp, flow_comp, arb_flow = nets
    with Image.open(src_paths[0]) as first_img:
        width, height = first_img.size
    if args.pad32:
        model_dim = (-(-width // 32) * 32, -(-height // 32) * 32)
    else:
        model_dim = (width // 32 * 32, height // 32 * 32)
    if model_dim[0] <= 0 or model_dim[1] <= 0:
        raise ValueError("invalid model_dim {} for {}".format(model_dim, rel))
    flow_back_warp = model.backWarp(model_dim[0], model_dim[1], device).to(device)

    t0 = time.time()

    prev_tensor, orig_dim = to_tensor(src_paths[resume_pairs], transform, model_dim, device, args.pad32)
    info_mode = "a" if out_idx > 0 else "w"
    with (out_images / "info.txt").open(info_mode, encoding="utf-8") as info:
        if out_idx == 0:
            with Image.open(src_paths[0]) as img0:
                out_path = output_frame_path(out_images, out_idx)
                save_pil(out_path, img0.convert("RGB"))
                save_info_line(info, out_path, out_idx, interp_fps)
                out_idx += 1

        with torch.no_grad():
            for pair_idx, curr_path in enumerate(src_paths[resume_pairs + 1 :], start=resume_pairs + 1):
                curr_tensor, _ = to_tensor(curr_path, transform, model_dim, device, args.pad32)
                mids = interpolate_pair_with_cudnn_fallback(
                    prev_tensor, curr_tensor, sf, flow_comp, arb_flow, flow_back_warp, rel, pair_idx
                )

                for ft in mids:
                    pil = tp(ft[0].cpu().detach())
                    if args.pad32:
                        pil = pil.crop((0, 0, orig_dim[0], orig_dim[1]))
                    else:
                        pil = pil.resize(orig_dim, Image.BILINEAR)
                    out_path = output_frame_path(out_images, out_idx)
                    save_pil(out_path, pil)
                    save_info_line(info, out_path, out_idx, interp_fps)
                    out_idx += 1

                with Image.open(curr_path) as src_img:
                    out_path = output_frame_path(out_images, out_idx)
                    save_pil(out_path, src_img.convert("RGB"))
                    save_info_line(info, out_path, out_idx, interp_fps)
                    out_idx += 1

                prev_tensor = curr_tensor

                if args.log_every > 0 and pair_idx % args.log_every == 0:
                    elapsed = time.time() - t0
                    done_this_run = pair_idx - resume_pairs
                    rate = done_this_run / elapsed if elapsed > 0 else 0.0
                    print(
                        "[{}] pair {}/{} ({:.3f} pair/s)".format(
                            rel, pair_idx, len(src_paths) - 1, rate
                        ),
                        flush=True,
                    )

    out_meta = dict(meta)
    out_meta.update(
        {
            "rel_path": str(rel),
            "sf": sf,
            "source_frames": len(src_paths),
            "interpolated_frames": out_idx,
            "interp_fps": interp_fps,
            "interp_image_pattern": "images/{:06d}.png",
        }
    )
    with (out_clip / "source_meta.json").open("w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)
        f.write("\n")

    return "ok", rel, out_idx, sf


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError("input_root not found: {}".format(input_root))
    if args.sf and args.sf < 2:
        raise ValueError("--sf must be >= 2, or 0 for auto")

    if args.manifest is not None:
        clips = collect_manifest_clips(input_root, args.manifest.resolve(), args.start_index, args.limit)
    else:
        clips = collect_clips(input_root, args.start_index, args.limit)
    print("[scan] clips={} input_root={}".format(len(clips), input_root), flush=True)
    print("[out]  output_root={}".format(output_root), flush=True)

    if args.dry_run:
        for clip in clips[:50]:
            with (clip / "source_meta.json").open("r", encoding="utf-8") as f:
                meta = json.load(f)
            src_paths = sorted((clip / "images").glob("*.png"))
            sf = calc_sf(float(meta.get("source_fps") or 0.0), args.sf, args.target_fps, meta.get("sf", 0))
            rel = clip.relative_to(input_root)
            print(
                "{} source_frames={} fps={} sf={} interp_frames={}".format(
                    rel, len(src_paths), meta.get("source_fps"), sf, expected_interp_frames(len(src_paths), sf)
                ),
                flush=True,
            )
        if len(clips) > 50:
            print("... {} more".format(len(clips) - 50), flush=True)
        return

    if not args.checkpoint.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(args.checkpoint))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[model] torch={} device={}".format(torch.__version__, device), flush=True)
    configure_cudnn(args, device)
    transform, tp = build_transforms(device)
    flow_comp = model.UNet(6, 4).to(device)
    arb_flow = model.UNet(20, 5).to(device)
    for param in flow_comp.parameters():
        param.requires_grad = False
    for param in arb_flow.parameters():
        param.requires_grad = False

    print("[model] loading checkpoint {}".format(args.checkpoint), flush=True)
    ckpt = load_checkpoint(args.checkpoint)
    flow_comp.load_state_dict(ckpt["state_dictFC"])
    arb_flow.load_state_dict(ckpt["state_dictAT"])
    nets = (device, transform, tp, flow_comp, arb_flow)

    counts = {}
    for i, clip in enumerate(clips):
        status, rel, frames, sf = process_clip(clip, input_root, output_root, args, nets)
        counts[status] = counts.get(status, 0) + 1
        print("[{}/{}] {} {} frames={} sf={}".format(i + 1, len(clips), status, rel, frames, sf), flush=True)

    print("[done] {}".format(counts), flush=True)


if __name__ == "__main__":
    main()
