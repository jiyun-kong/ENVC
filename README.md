# ENVC

Anonymous repository for the ICLR 2027 submission "Event-guided Neural Video Compression".

## Setup

The supported environment uses Python 3.10, PyTorch 2.11.0 with CUDA 13.0, and CuPy 14.1.1.

```bash
conda create -n ENVC python=3.10
conda activate ENVC
pip install -r requirements.txt
```

## Datasets

ENVC is evaluated on the first 96 frames of UVG, MCL-JCV, and HEVC Class B, C, D, and E. Each test set lives in its own folder, with the RGB frames as 8-bit YUV 4:2:0 files and the synthetic events next to them:

```
ENVC_testsets/
├── UVG/
│   ├── yuv/
│   │   ├── Beauty_1920x1080_120.yuv
│   │   └── ...
│   └── events/
│       ├── Beauty_1920x1080_120/
│       │   ├── 000000.npz        # events between frames 0 and 1
│       │   ├── ...
│       │   └── 000094.npz        # events between frames 94 and 95
│       └── ...
├── MCL-JCV/
├── HEVC-B/
├── HEVC-C/
├── HEVC-D/
└── HEVC-E/
```

- **Events.** Download the synthetic events used in the paper from [Google Drive](https://drive.google.com/file/d/1N5y1z8BqDJxVxFGXmkqcgFFypo9rCe3p/view?usp=sharing) (`ENVC_testsets.zip`) and unzip it. This creates the folder tree above with every `events/` folder filled in. Each `.npz` file holds the arrays `x`, `y`, `timestamp` (µs), and `polarity`. File `k` covers the interval from frame `k` to frame `k+1`, so a 96-frame sequence has 95 files.
- **RGB frames.** Get the original sequences from their official sources ([UVG](https://ultravideo.fi/dataset.html), [MCL-JCV](https://mcl.usc.edu/mcl-jcv-dataset/), and the JCT-VC HEVC common test conditions). Then cut them to 96 frames and give them the names the events use:

  ```bash
  python eval/prepare_yuv.py --src /path/to/official/UVG --out ENVC_testsets/UVG/yuv
  ```

  Files are named `<name>_<W>x<H>_<fps>.yuv`. For example, `Beauty_1920x1080_120fps_420_8bit_YUV.yuv` becomes `Beauty_1920x1080_120.yuv`. The frame size and frame rate are read from the name. The evaluation converts YUV to RGB with BT.709 (full range, bilinear chroma upsampling), as in DCVC-DC and DCMVC.

Then pass `--dataset ENVC_testsets/<test set> --layout yuv` to the evaluation below. Events are read from `<dataset>/events/<sequence>/` by default; `--event-root` selects another location.

### Generating Synthetic Events

[`event_simulation/`](event_simulation/) contains the pipeline that produced the released events. It runs three steps:

1. `make_frames.py` decodes the YUV files to RGB with the same BT.709 conversion as the evaluation.
2. `interpolate.py` upsamples the frame rate to at least 200 fps with [Super-SloMo](https://github.com/avinashpaliwal/Super-SloMo), using a factor of ⌈200 / fps⌉ and reflect padding to a multiple of 32.
3. `simulate_events.py` runs the [DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter) simulator with DVS346 parameters and merges the events of each original frame interval into one file.

Download the pretrained `SuperSloMo.ckpt` from the [Super-SloMo repository](https://github.com/avinashpaliwal/Super-SloMo), install the extra packages, and run:

```bash
pip install -r event_simulation/requirements.txt
CKPT=/path/to/SuperSloMo.ckpt DATASET_DIR=ENVC_testsets/HEVC-D bash event_simulation/generate_events.sh
```

This writes `ENVC_testsets/HEVC-D/events/<sequence>/*.npz` from `ENVC_testsets/HEVC-D/yuv/`. Super-SloMo uses a GPU when one is visible. The simulator is stochastic, so regenerated events are statistically equivalent but not bit-identical to the released ones; use the released events to reproduce the reported numbers exactly. The Super-SloMo model and the DVS-Voltmeter simulator are included under `event_simulation/third_party/` with their MIT licenses. The only change is `align_corners=True` in Super-SloMo's warping, which restores the behavior of the PyTorch version it was trained with.

## Evaluation

Run inside a GPU allocation or on a GPU server:

```bash
bash eval/EVAL.sh \
  --dataset ENVC_testsets/HEVC-D --layout yuv \
  --p-frame-model-path /path/to/envc_weights.pt \
  --i-frame-model-path /path/to/intra_weights.pt \
  --quality 0 --num-frames 96 --intra-period 32 --resume
```

For the four embedded rate points:

```bash
for quality in 0 1 2 3; do
  bash eval/EVAL.sh \
    --dataset ENVC_testsets/HEVC-D --layout yuv \
    --p-frame-model-path /path/to/envc_weights.pt \
    --i-frame-model-path /path/to/intra_weights.pt \
    --quality "$quality" --num-frames 96 --intra-period 32 --resume || break
done
```

## ENVC-L (Lightweight Model)

ENVC-L is a lightweight variant of ENVC with reduced computational cost. It keeps the ENVC architecture and changes two parts:

- a narrower decoder in the event-conditioned frame predictor (decoder widths 160/128/128/96 instead of 320/256/192/128);
- prediction-guided context fusion computed at half resolution, with the learned correction upsampled to full resolution.

Both models share the same code (`models/versions.py`). Select ENVC-L with `--model envc-l` and its own P-frame checkpoint; the default `--model envc` is the full model.

```bash
bash eval/EVAL.sh --model envc-l \
  --dataset ENVC_testsets/HEVC-D --layout yuv \
  --p-frame-model-path /path/to/envc_l_weights.pt \
  --i-frame-model-path /path/to/intra_weights.pt \
  --quality 0 --num-frames 96 --intra-period 32 --resume
```

## Model Checkpoints

Each model needs its P-frame checkpoint and the shared I-frame checkpoint:

| Model | Checkpoint | Argument |
|---|---|---|
| ENVC (P-frame) | [Google Drive](https://drive.google.com/file/d/1M4Bark80TUKJAbsHuFaRUt0xqWU5s5UW/view?usp=sharing) | `--p-frame-model-path` |
| ENVC-L (P-frame, lightweight) | [Google Drive](https://drive.google.com/file/d/1MQOE6YhBrUMQaWdqOs455FwyMPUCW2rc/view?usp=sharing) | `--p-frame-model-path` (with `--model envc-l`) |
| DCVC-DC image model (I-frame) | `cvpr2023_image_psnr.pth.tar` from the [official DCVC-DC repository](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-DC) | `--i-frame-model-path` |

## Acknowledgments

The implementation includes components derived from DCVC, DCMVC, and VFPSIE, and the event simulation uses Super-SloMo and DVS-Voltmeter. Existing source copyright and license notices are retained. See [LICENSE](LICENSE) and [licenses/](licenses/).

## Per-frame Results

[`results/`](results/) lists the per-frame rate and quality of ENVC and the anchors (DCVC, DCVC-DC, DCMVC, DCVC-FM, DCVC-RT), one file per test set: [`UVG.csv`](results/UVG.csv), [`MCL-JCV.csv`](results/MCL-JCV.csv), [`HEVC-B.csv`](results/HEVC-B.csv), [`HEVC-C.csv`](results/HEVC-C.csv), [`HEVC-D.csv`](results/HEVC-D.csv), [`HEVC-E.csv`](results/HEVC-E.csv). Each covers the first 96 frames of every sequence, intra period 32, and four rate points per method, with RGB frames converted from the YUV sources. All files share the columns below.

| Column | Meaning |
|---|---|
| `dataset`, `sequence` | test set and sequence |
| `method` | codec |
| `rate_point` | 0 (lowest rate) to 3 (highest rate); for DCVC-FM and DCVC-RT these are q_index 0, 21, 42, 63 |
| `frame`, `frame_type` | frame index (0 to 95) and `I` / `P` (I-frames at 0, 32, 64) |
| `bpp` | bits per pixel of the frame |
| `psnr` | PSNR-RGB (dB) |
| `psnr_y`, `psnr_u`, `psnr_v`, `psnr_yuv` | PSNR of the Y, U, V planes and the 6:1:1 weighted YUV PSNR (dB) |
| `ssim`, `ms_ssim`, `lpips` | SSIM, MS-SSIM (RGB), LPIPS (AlexNet) |
