# ENVC

Anonymous repository for the ICLR 2027 submission "Event-guided Neural Video Compression".

## Setup

The supported environment uses Python 3.10, PyTorch 2.11.0 with CUDA 13.0, and CuPy 14.1.1.

```bash
conda create -n ENVC python=3.10
conda activate ENVC
pip install -r requirements.txt
```

## Evaluation

Run inside a GPU allocation or on a GPU server:

```bash
bash eval/EVAL.sh \
  --dataset /path/to/HEVC-D --layout yuv \
  --p-frame-model-path /path/to/envc_weights.pt \
  --i-frame-model-path /path/to/intra_weights.pt \
  --quality 0 --num-frames 96 --intra-period 32 --resume
```

For the four embedded rate points:

```bash
for quality in 0 1 2 3; do
  bash eval/EVAL.sh \
    --dataset /path/to/HEVC-D --layout yuv \
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
  --dataset /path/to/HEVC-D --layout yuv \
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

The implementation includes components derived from DCVC, DCMVC, and VFPSIE. Existing source copyright and license notices are retained. See [LICENSE](LICENSE) and [licenses/](licenses/).

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
