# src/eval.py

import argparse
import os
from pathlib import Path

import torch
import torchaudio

from src.utils.metrics import snr_db, mse, mae, log_spectral_distance, compute_stoi_pesq


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def safe_mean(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and (v != v))]
    if not clean:
        return float("nan")
    return float(torch.tensor(clean).mean().item())


def main(args):
    hr_dir = Path(args.hr_dir)
    sr_dir = Path(args.sr_dir)

    if not hr_dir.is_dir():
        raise RuntimeError(f"HR dir not found: {hr_dir}")
    if not sr_dir.is_dir():
        raise RuntimeError(f"SR dir not found: {sr_dir}")

    # Collect SR files
    sr_files = [
        f for f in sr_dir.iterdir()
        if f.suffix.lower() in [".wav", ".flac", ".ogg"]
    ]
    if not sr_files:
        raise RuntimeError(f"No SR audio files found in {sr_dir}")

    print(f"Found {len(sr_files)} SR files in {sr_dir}")
    print("Matching against HR files in", hr_dir)

    metrics = {
        "snr": [],
        "mse": [],
        "mae": [],
        "lsd": [],
        "stoi": [],
        "pesq": [],
    }

    for sr_path in sorted(sr_files):
        base = sr_path.stem  # name without extension
        hr_candidate = None
        for ext in [".wav", ".flac", ".ogg"]:
            cand = hr_dir / (base + ext)
            if cand.is_file():
                hr_candidate = cand
                break

        if hr_candidate is None:
            print(f"[WARN] No HR match for {sr_path.name}, skipping.")
            continue

        wav_sr = load_audio(str(sr_path), args.sample_rate)
        wav_hr = load_audio(str(hr_candidate), args.sample_rate)

        # Align
        min_len = min(wav_hr.shape[-1], wav_sr.shape[-1])
        wav_hr = wav_hr[..., :min_len]
        wav_sr = wav_sr[..., :min_len]

        # Metrics
        snr_val = snr_db(wav_hr, wav_sr)
        mse_val = mse(wav_hr, wav_sr)
        mae_val = mae(wav_hr, wav_sr)
        lsd_val = log_spectral_distance(wav_hr, wav_sr)
        stoi_val, pesq_val = compute_stoi_pesq(wav_hr, wav_sr, args.sample_rate)

        metrics["snr"].append(snr_val)
        metrics["mse"].append(mse_val)
        metrics["mae"].append(mae_val)
        metrics["lsd"].append(lsd_val)
        metrics["stoi"].append(stoi_val)
        metrics["pesq"].append(pesq_val)

        print(
            f"{sr_path.name} | "
            f"SNR={snr_val:.2f} dB, LSD={lsd_val:.3f}, "
            f"MSE={mse_val:.6f}, MAE={mae_val:.6f}"
        )

    print("\n=== AVERAGE METRICS OVER SR DIR ===")
    print(f"SNR (dB): {safe_mean(metrics['snr']):.4f}")
    print(f"MSE:      {safe_mean(metrics['mse']):.6f}")
    print(f"MAE:      {safe_mean(metrics['mae']):.6f}")
    print(f"LSD:      {safe_mean(metrics['lsd']):.6f}")
    print(f"STOI:     {safe_mean(metrics['stoi']):.4f}")
    print(f"PESQ:     {safe_mean(metrics['pesq']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SR WAV files against HR ground truth."
    )
    parser.add_argument("--hr_dir", type=str, required=True,
                        help="Directory with HR reference files (e.g., data/hr)")
    parser.add_argument("--sr_dir", type=str, required=True,
                        help="Directory with SR output files (e.g., outputs/)")
    parser.add_argument("--sample_rate", type=int, default=48000)
    args = parser.parse_args()
    main(args)
