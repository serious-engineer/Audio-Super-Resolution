# Speech Bandwidth Extension (SBE) — CNN + GAN

This repo is a **starter kit** for your EEE607 project to enhance 8 kHz narrowband speech to 16 kHz wideband using a CNN generator and a GAN discriminator. It includes:
- Data prep scripts to make NB/WB pairs (WB → LPF + downsample to 8 kHz → upsample back to 16 kHz for baseline).
- A baseline upsampler (interpolation only).
- CNN+GAN scaffolding (PyTorch) with PatchGAN-style discriminator on spectrograms.
- Evaluation scripts (PESQ, STOI proxy for intelligibility, MOS/DRT templates).

> You’ll need to drop some **wideband** WAVs (16 kHz) into `data/raw_wb/` to get started.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 1) Make NB/WB pairs and a tiny manifest
python src/utils/prepare_pairs.py --in_dir data/raw_wb --out_dir data/processed_nbwb --sr_wb 16000 --sr_nb 8000

# 2) Run baseline (interpolation only) for a sanity check and metrics
python src/baselines/run_baseline.py --pairs data/processed_nbwb/manifest.csv --out_dir experiments/baseline_out

# 3) Train CNN+GAN (spectrogram domain)
python src/trainers/train_cnn_gan.py --config configs/cnn_gan_mel.yaml
```

## Folder layout

- `data/raw_wb/` — put 16 kHz clean/wideband WAV here (mono).
- `data/processed_nbwb/` — auto-generated NB/WB pairs + CSV manifest.
- `src/baselines/` — interpolation baseline.
- `src/models/` — generator and discriminator definitions.
- `src/trainers/` — training loop for GAN.
- `src/eval/` — PESQ/STOI evaluation + MOS/DRT templates.
- `configs/` — YAML config(s).
- `experiments/` — outputs (checkpoints, samples, logs).

## Notes

- PESQ is the official ITU-T metric for narrowband (8 kHz) and wideband (16 kHz). We compute PESQ on **time-aligned** WB references.
- MOS and DRT require listeners. We include a tiny HTML + CSV scaffolding to run AB tests locally and log ratings.
