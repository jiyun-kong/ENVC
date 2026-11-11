#!/usr/bin/env python3
"""Evaluate ENVC / ENVC-L using entropy-estimated rates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from PIL import Image

from models import create_model
from models.image_model import IntraNoAR
from utils.envc_stream import get_padding_size
from utils.event_voxel import NUM_BINS, build_voxel
from utils.metrics import (
    compute_psnr, compute_psnr_y, compute_ssim,
    compute_ms_ssim, compute_lpips, avg_frame_metrics,
)

ENVC_MIN_DIV = 64


# ----------------------------------------------------------------------------
# Model loading, event inputs, and per-sequence coding
# ----------------------------------------------------------------------------

def _load_state_dict(ckpt_path: str) -> dict:
    """Load an ENVC/DCVC-style checkpoint into a plain state dict."""
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict):
        for key in ("state_dict", "net", "model", "model_state_dict", "p_net"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                ckpt = value
                break

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint {ckpt_path} did not contain a PyTorch state dict."
        )

    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt

def _require_state_keys(state: dict, required: set[str], ckpt_path: str, label: str) -> None:
    missing = sorted(required.difference(state.keys()))
    if not missing:
        return

    sample = ", ".join(list(state.keys())[:8])
    raise RuntimeError(
        f"{label} checkpoint does not look compatible with ENVC.\n"
        f"Path: {ckpt_path}\n"
        f"Missing keys: {', '.join(missing)}\n"
        f"First checkpoint keys: {sample}"
    )

def load_i_frame_net(device: torch.device, ckpt_path: str):
    """Load the IntraNoAR I-frame model (frozen DCVC-DC intra)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"I-frame checkpoint not found: {ckpt_path}")
    state = _load_state_dict(ckpt_path)
    _require_state_keys(state, {"q_scale_enc", "q_scale_dec"}, ckpt_path, "I-frame")
    net = IntraNoAR(inplace=True)
    net.load_state_dict(state)
    net = net.to(device)
    net.eval()
    return net, ckpt_path


def load_p_frame_net(device, ckpt_path, model_version='envc'):
    state = _load_state_dict(ckpt_path)
    _require_state_keys(state, {
        "y_q_scale_enc", "y_q_scale_dec", "mv_y_q_scale_enc", "mv_y_q_scale_dec",
        "front_end.image_encoder.pyramid1.0.0.weight",
    }, ckpt_path, "ENVC P-frame")
    net = create_model(model_version, inplace=True)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval(), str(ckpt_path)


def event_path(npz_dir, interval_idx):
    candidates = [Path(npz_dir) / f"{interval_idx:05d}.npz",
                  Path(npz_dir) / f"{interval_idx:06d}.npz"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one event interval {interval_idx} in {npz_dir}; found {found}")
    return found[0]


def load_event_voxel(npz_dir, interval_idx, width, height, device):
    path = event_path(npz_dir, interval_idx)
    with np.load(path) as data:
        n_events = int(data["x"].shape[0])
    voxel = build_voxel(path, width, height, NUM_BINS)
    return torch.from_numpy(voxel).unsqueeze(0).to(device), n_events / float(width * height)


def compress_sequence(
    p_net,
    i_net,
    frames: list,
    npz_dir: str,
    i_q_index: int,
    p_q_index: int,
    device: torch.device,
    width: int,
    height: int,
    intra_period: int = -1,
    lpips_fn=None,
):
    """
    Compress and reconstruct a list of RGB frames using ENVC.

    intra_period = -1 : single I-frame at frame 0, rest are P-frames.
    intra_period = N  : I-frame every N frames.

    q_in_ckpt=True for the four embedded rate points; frame_idx % 4 keeps the
    P-frame model's context adaptor on the four-frame context cycle. P-frame t uses
    the event voxel of interval t-1 -> t (npz file t-1).
    """
    if not frames:
        return []
    if intra_period == 0 or intra_period < -1:
        raise ValueError("intra_period must be -1 or a positive integer.")

    h, w = frames[0].shape[1], frames[0].shape[2]
    pixels = h * w
    pad_l, pad_r, pad_t, pad_b = get_padding_size(h, w, ENVC_MIN_DIV)

    frame_results: list[dict] = []
    dpb = None

    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            x = frame.unsqueeze(0).to(device)  # (1, C, H, W), float32 [0, 1]
            x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")

            is_intra = (frame_idx == 0) or (
                intra_period > 0 and frame_idx % intra_period == 0
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if is_intra:
                result = i_net.encode_decode(
                    x_pad,
                    q_in_ckpt=True,
                    q_index=i_q_index,
                    output_path=None,
                    pic_height=h,
                    pic_width=w,
                )
                rec_pad = result["x_hat"].clamp(0.0, 1.0)
                bpp = result["bit"] / pixels
                bpp_parts = {}

                dpb = {
                    "ref_frame": rec_pad,
                    "ref_feature": None,
                    "ref_y": None,
                }
            else:
                if dpb is None:
                    raise RuntimeError("ENVC P-frame encountered before DPB initialisation.")
                voxel, events_per_pixel = load_event_voxel(
                    npz_dir, frame_idx - 1, w, h, device
                )
                # Events outside the frame do not exist: zero-pad, never
                # replicate (replication would fabricate border events).
                voxel_pad = F.pad(voxel, (pad_l, pad_r, pad_t, pad_b),
                                  mode="constant", value=0.0)
                result = p_net.encode_decode(
                    x_pad,
                    dpb,
                    voxel_pad,
                    q_in_ckpt=True,
                    q_index=p_q_index,
                    output_path=None,
                    pic_height=h,
                    pic_width=w,
                    frame_idx=frame_idx % 4,
                )
                dpb = result["dpb"]
                rec_pad = dpb["ref_frame"].clamp(0.0, 1.0)
                bpp = result["bit"] / pixels
                # Separate transmitted residual-motion and image-residual rates.
                bpp_parts = {
                    "bpp_mv": round((result["bit_mv_y"] + result["bit_mv_z"]) / pixels, 6),
                    "bpp_res": round((result["bit_y"] + result["bit_z"]) / pixels, 6),
                    "events_per_pixel": round(events_per_pixel, 6),
                }
                del voxel, voxel_pad

            del x_pad, x

            rec = rec_pad[:, :, pad_t:pad_t + h, pad_l:pad_l + w]
            orig_b = frame.unsqueeze(0)
            rec_b = rec.cpu()

            psnr = compute_psnr(orig_b, rec_b)
            psnr_y = compute_psnr_y(orig_b, rec_b)
            ssim = compute_ssim(orig_b, rec_b)
            ms_ssim = compute_ms_ssim(orig_b, rec_b)
            lp = compute_lpips(orig_b.to(device), rec_b.to(device), lpips_fn)

            frame_dict: dict = {
                "frame_idx": frame_idx,
                "is_intra": is_intra,
                "psnr": round(psnr, 4),
                "psnr_y": round(psnr_y, 4),
                "bpp": round(bpp, 6),
                **bpp_parts,
            }
            if ssim is not None:
                frame_dict["ssim"] = round(ssim, 6)
            if ms_ssim is not None:
                frame_dict["ms_ssim"] = round(ms_ssim, 6)
            if lp is not None:
                frame_dict["lpips"] = round(lp, 6)

            frame_results.append(frame_dict)

    return frame_results

def load_png_frames(paths: list[str]) -> tuple[list[torch.Tensor], int, int]:
    """CHW float32 RGB tensors in [0, 1], matching yuv_io.read_yuv_frames()'s
    output contract exactly, so compress_sequence() needs no changes."""
    frames = []
    width = height = None
    for path in paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if width is None:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError(f"{path} is {w}x{h}, expected {width}x{height}")
        arr = torch.from_numpy(
            np.array(img, dtype="float32") / 255.0
        )
        frames.append(arr.permute(2, 0, 1).contiguous())
    return frames, width, height


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--layout', choices=('native', 'pairs', 'sequences', 'yuv'), default='native',
                        help='Explicit frame protocol; native=color PNGs, pairs=simulator frames, sequences=images/events, yuv=baseline RGB conversion')
    parser.add_argument('--frames-subdir', default='frames_color_native')
    parser.add_argument('--seq-subdir', default='sequences')
    parser.add_argument('--event-root', type=Path)
    parser.add_argument('--event-subdir', default='')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sequences', nargs='+')
    parser.add_argument('--quality', type=int, choices=range(4), default=1)
    parser.add_argument('--num-frames', type=int, default=96, help='<=0: all frames')
    parser.add_argument('--intra-period', type=int, default=32)
    parser.add_argument('--model', choices=('envc', 'envc-l'), default='envc',
                        help='envc: full model; envc-l: lightweight model (use its own checkpoint)')
    parser.add_argument('--p-frame-model-path', type=Path, required=True)
    parser.add_argument('--i-frame-model-path', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, help='Default: results/<model>/<dataset>/<layout>/q<quality>-ip<period>')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-inputs', action='store_true', help='Check frames, event intervals, and weight paths without inference')
    args = parser.parse_args(argv)
    if args.intra_period == 0 or args.intra_period < -1:
        parser.error('--intra-period must be -1 or positive')
    if args.threads < 1:
        parser.error('--threads must be positive')
    if (args.width is None) != (args.height is None):
        parser.error('--width and --height must be supplied together')
    if args.width is not None and min(args.width, args.height) < 1:
        parser.error('Dimensions must be positive')
    if args.resume and (args.layout != 'yuv' or args.num_frames < 1):
        parser.error('--resume requires YUV input and a positive frame count')
    from utils.datasets import is_mcl_jcv, resolve_eval_dataset
    args.dataset = resolve_eval_dataset(args.dataset, args.width, args.height).resolve()
    args.dataset_name = "MCL-JCV" if is_mcl_jcv(args.dataset) else args.dataset.name
    if args.save_dir is None:
        args.save_dir = ROOT/'results'/args.model/args.dataset_name/args.layout/f'q{args.quality}-ip{args.intra_period}'
    return args


def collect_inputs(args):
    dataset = args.dataset
    folder = dataset / {'native': args.frames_subdir, 'pairs': 'pairs',
                        'sequences': args.seq_subdir, 'yuv': 'yuv'}[args.layout]
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing {args.layout} input directory: {folder}; select the intended --layout explicitly')
    if args.layout == 'yuv':
        entries = {p.stem: p for p in sorted(folder.glob('*.yuv')) if p.is_file()}
    else:
        entries = {p.name: p for p in sorted(folder.iterdir()) if p.is_dir()}
    if args.sequences:
        missing = set(args.sequences) - entries.keys()
        if missing:
            raise ValueError(f'Unknown sequences: {sorted(missing)}')
        entries = {name: entries[name] for name in dict.fromkeys(args.sequences)}
    if not entries:
        raise ValueError(f'No input sequences in {folder}')
    inputs = []
    for name, entry in entries.items():
        width = height = None
        if args.layout == 'yuv':
            match = re.search(r'(\d+)x(\d+)', name)
            if args.width is not None:
                width, height = args.width, args.height
                if match and (width, height) != tuple(map(int, match.groups())):
                    raise ValueError(f'Explicit dimensions disagree with {entry}')
            elif match:
                width, height = map(int, match.groups())
            else:
                raise ValueError(f'Missing WxH in {entry}; pass --width and --height')
            if width % 2 or height % 2:
                raise ValueError(f'YUV420 requires even dimensions: {entry}')
            count, remainder = divmod(entry.stat().st_size, width * height * 3 // 2)
            if remainder:
                raise ValueError(f'Invalid YUV420 file size: {entry}')
            paths = entry
        else:
            if args.layout == 'pairs':
                paths = [dataset/'frames'/name/Path(line.split('\t')[0]).name
                         for line in (entry/'pairs.txt').read_text().splitlines()
                         if line.strip() and not line.startswith('#')]
            else:
                paths = sorted((entry/'images' if args.layout == 'sequences' else entry).glob('*.png'))
            count = len(paths)
        if count < 1 or (args.num_frames > 0 and count < args.num_frames):
            raise ValueError(f'{name}: {count} frames available, requested {args.num_frames}')
        count = min(count, args.num_frames) if args.num_frames > 0 else count
        if isinstance(paths, list):
            paths = paths[:count]
            for p in paths:
                if not p.is_file():
                    raise FileNotFoundError(p)
        events = (args.event_root/name/args.event_subdir if args.event_root else
                  entry/'events' if args.layout == 'sequences' else dataset/'event_voxel_npz'/name)
        for index in range(1, count):
            if args.intra_period < 0 or index % args.intra_period:
                event_path(events, index - 1)
        inputs.append((name, paths, events, count, width, height))
    for p in (args.i_frame_model_path, args.p_frame_model_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    return inputs


def main(argv=None):
    args = parse_args(argv)
    inputs = collect_inputs(args)
    for name, _, events, count, _, _ in inputs:
        print(f'[INPUT] {name}: {count} frames, events={events}', flush=True)
    if args.check_inputs:
        print('Input paths and required event intervals OK; model inference was not run.')
        return

    import cupy
    from utils.metrics import init_lpips, warn_missing_libs
    from utils.utils import set_seed
    from utils.yuv_io import read_yuv_frames
    from utils.eval_resume import (atomic_json, prepare_run, sequence_provenance,
                                   load_cached_sequence, valid_sequence)

    if not torch.cuda.is_available():
        raise RuntimeError('ENVC softsplat requires CUDA. Run from your allocated GPU shell.')
    if not hasattr(cupy.cuda, 'compile_with_cache'):
        raise RuntimeError(f'The softsplat compiler is unavailable in CuPy {cupy.__version__}. Check the documented CUDA/CuPy dependencies.')
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    i_net, _ = load_i_frame_net(device, args.i_frame_model_path)
    p_net, _ = load_p_frame_net(device, args.p_frame_model_path, args.model)
    warn_missing_libs()
    lpips_fn = init_lpips(device)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(model='ENVC-L' if args.model == 'envc-l' else 'ENVC', rate_mode='entropy_estimate',
                    event_bits_included=False, feature_reset_threshold=5.0,
                    deterministic_softsplat=os.environ.get('EVCOM_DETERMINISTIC_SOFTSPLAT', '0'),
                    deterministic_softsplat_block=os.environ.get('EVCOM_DETERMINISTIC_SOFTSPLAT_BLOCK', '32'))
    signed = args.layout == 'yuv' and args.num_frames > 0
    run_args = None
    if signed:
        dimensions = {(row[4], row[5]) for row in inputs}
        if len(dimensions) != 1:
            raise ValueError('A signed dataset evaluation requires uniform dimensions')
        width, height = dimensions.pop()
        event_inputs = {}
        for name, _, events, count, _, _ in inputs:
            intervals = []
            for index in range(1, count):
                if args.intra_period > 0 and index % args.intra_period == 0:
                    continue
                path = event_path(events, index - 1)
                stat = path.stat()
                intervals.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
            event_inputs[name] = intervals
        run_args = argparse.Namespace(**metadata)
        run_args.width, run_args.height = width, height
        run_args.eval_profile = 'rgb_bt709_full_bilinear_8bit'
        run_args.event_inputs = event_inputs
        provenance = prepare_run(run_args, 'envc',
                                 [args.p_frame_model_path, args.i_frame_model_path],
                                 'entropy_estimate', source_roots=(ROOT/'models', ROOT/'utils', ROOT/'eval'))
        metadata.update(provenance, width=width, height=height,
                        num_frames_requested=args.num_frames, sequence_filter=args.sequences)
    atomic_json(args.save_dir/'args.json', metadata)
    summaries = {}
    for name, paths, events, count, width, height in inputs:
        cached = load_cached_sequence(run_args, paths, name) if signed else None
        if cached is not None:
            summaries[name] = {key: value for key, value in cached.items() if key.startswith('avg_')}
            print(f'[RESUME] {name}', flush=True)
            continue
        print(f'[EVAL] {name} q{args.quality}', flush=True)
        if args.layout == 'yuv':
            frames = read_yuv_frames(str(paths), width, height, count)
        else:
            frames, width, height = load_png_frames(paths)
        metrics = compress_sequence(p_net, i_net, frames, str(events), args.quality, args.quality,
                                    device, width, height, args.intra_period, lpips_fn=lpips_fn)
        averages = {f'avg_{key}': value for key, value in avg_frame_metrics(metrics).items()}
        result = dict(metadata, sequence=name, width=width, height=height,
                      num_frames_coded=len(metrics), num_intra_frames=sum(r['is_intra'] for r in metrics),
                      **averages, frames=metrics)
        if signed:
            provenance = sequence_provenance(run_args, paths)
            result.update(provenance)
            if not valid_sequence(result, run_args, provenance):
                raise RuntimeError(f'{name}: incomplete or invalid evaluation metrics')
        atomic_json(args.save_dir/f'{name}.json', result)
        summaries[name] = averages
        del frames
        print(f"[DONE] {name}: PSNR={averages['avg_psnr']:.4f}, bpp={averages['avg_bpp']:.6f}", flush=True)
    summary_name = 'summary.partial.json' if args.sequences else 'summary.json'
    means = {key.replace('avg_', 'mean_', 1): sum(row[key] for row in summaries.values()) / len(summaries)
             for key in next(iter(summaries.values()))}
    atomic_json(args.save_dir/summary_name,
                dict(metadata, num_sequences=len(summaries), sequences=summaries, **means))



if __name__ == '__main__':
    main()
