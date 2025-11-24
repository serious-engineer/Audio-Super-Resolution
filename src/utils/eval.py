# src/eval.py
import argparse
from pathlib import Path

import torch
import torchaudio
import matplotlib.pyplot as plt

from src.utils.metrics import (
    snr_db,
    mse,
    mae,
    log_spectral_distance,
    compute_stoi_pesq,
)

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


def find_hr_lr_match(base: str, hr_dir: Path, low_dir: Path):
    """
    Match SR filename to HR and LR filenames, given naming convention below
        SR: p225_sr.wav
        HR: p225.wav
        LR: p225.wav
    """
    # Strip trailing "_sr" if present
    if base.endswith("_sr"):
        core = base[:-3]
    else:
        core = base

    hr_path = None
    lr_path = None

    for ext in [".wav", ".flac", ".ogg"]:
        cand_hr = hr_dir / f"{core}{ext}"
        if cand_hr.is_file():
            hr_path = cand_hr
            break

    for ext in [".wav", ".flac", ".ogg"]:
        cand_lr = low_dir / f"{core}{ext}"
        if cand_lr.is_file():
            lr_path = cand_lr
            break

    return hr_path, lr_path


def save_spectrogram_triple(
    wav_hr: torch.Tensor,
    wav_lr: torch.Tensor,
    wav_sr: torch.Tensor,
    sample_rate: int,
    out_path: Path,
    n_fft: int = 1024,
    hop_length: int = 256,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hr_1d = wav_hr.squeeze(0)
    lr_1d = wav_lr.squeeze(0)
    sr_1d = wav_sr.squeeze(0)

    def stft_logmag(x):
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            return_complex=True,
        )
        mag = spec.abs()
        eps = 1e-8
        return 20.0 * torch.log10(mag + eps)

    log_hr = stft_logmag(hr_1d)
    log_lr = stft_logmag(lr_1d)
    log_sr = stft_logmag(sr_1d)

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.imshow(
        log_hr.cpu().numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.title(f"HR Spectrogram ({sample_rate} Hz)")
    plt.ylabel("Freq bins")

    plt.subplot(3, 1, 2)
    plt.imshow(
        log_lr.cpu().numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.title("LR Spectrogram (resampled to eval SR)")
    plt.ylabel("Freq bins")

    plt.subplot(3, 1, 3)
    plt.imshow(
        log_sr.cpu().numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.title("SR Spectrogram")
    plt.xlabel("Time frames")
    plt.ylabel("Freq bins")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    hr_dir = Path(args.hr_dir)
    low_dir = Path(args.low_sr_dir)

    if not hr_dir.is_dir():
        raise RuntimeError(f"HR dir not found: {hr_dir}")
    if not low_dir.is_dir():
        raise RuntimeError(f"Low-SR dir not found: {low_dir}")

    repo_root = Path(__file__).resolve().parent.parent
    metrics_root = repo_root / "metrics"
    specs_dir = metrics_root / "spectrograms"
    specs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Spectrograms will be saved to: {specs_dir}")

    if args.sr_file:
        sr_dir = None
        sr_files = [Path(args.sr_file)]
        print(f"[Single file mode] Evaluating SR file: {args.sr_file}")
    else:
        sr_dir = Path(args.sr_dir)
        if not sr_dir.is_dir():
            raise RuntimeError(f"SR dir not found: {sr_dir}")

        sr_files = [
            f for f in sr_dir.iterdir()
            if f.suffix.lower() in [".wav", ".flac", ".ogg"]
        ]
        if not sr_files:
            raise RuntimeError(f"No SR audio files found in {sr_dir}")

        print(f"Found {len(sr_files)} SR files in {sr_dir}")
        print("HR files from:", hr_dir)
        print("LR files from:", low_dir)

    metrics_hr_sr = {k: [] for k in ["snr", "mse", "mae", "lsd", "stoi", "pesq"]}
    metrics_hr_lr = {k: [] for k in ["snr", "mse", "mae", "lsd", "stoi", "pesq"]}
    metrics_lr_sr = {k: [] for k in ["snr", "mse", "mae", "lsd", "stoi", "pesq"]}

    for sr_path in sorted(sr_files):
        base = sr_path.stem 

        hr_path, lr_path = find_hr_lr_match(base, hr_dir, low_dir)
        if hr_path is None:
            print(f"[WARN] No HR match for {sr_path.name}, skipping.")
            continue
        if lr_path is None:
            print(f"[WARN] No LR match for {sr_path.name}, LR-based metrics will be skipped.")

        wav_sr = load_audio(str(sr_path), args.sample_rate)   
        wav_hr = load_audio(str(hr_path), args.sample_rate)  

        if lr_path is not None:
            wav_lr = load_audio(str(lr_path), args.sample_rate)  
        else:
            wav_lr = torch.zeros_like(wav_hr)

        min_len = min(wav_hr.shape[-1], wav_sr.shape[-1], wav_lr.shape[-1])
        wav_hr = wav_hr[..., :min_len]
        wav_sr = wav_sr[..., :min_len]
        wav_lr = wav_lr[..., :min_len]

        # HR vs SR metrics
        snr_hr_sr = snr_db(wav_hr, wav_sr)
        mse_hr_sr = mse(wav_hr, wav_sr)
        mae_hr_sr = mae(wav_hr, wav_sr)
        lsd_hr_sr = log_spectral_distance(wav_hr, wav_sr)
        stoi_hr_sr, pesq_hr_sr = compute_stoi_pesq(wav_hr, wav_sr, args.sample_rate)

        metrics_hr_sr["snr"].append(snr_hr_sr)
        metrics_hr_sr["mse"].append(mse_hr_sr)
        metrics_hr_sr["mae"].append(mae_hr_sr)
        metrics_hr_sr["lsd"].append(lsd_hr_sr)
        metrics_hr_sr["stoi"].append(stoi_hr_sr)
        metrics_hr_sr["pesq"].append(pesq_hr_sr)

        # HR vs LR metrics
        if lr_path is not None:
            snr_hr_lr = snr_db(wav_hr, wav_lr)
            mse_hr_lr = mse(wav_hr, wav_lr)
            mae_hr_lr = mae(wav_hr, wav_lr)
            lsd_hr_lr = log_spectral_distance(wav_hr, wav_lr)
            stoi_hr_lr, pesq_hr_lr = compute_stoi_pesq(wav_hr, wav_lr, args.sample_rate)

            metrics_hr_lr["snr"].append(snr_hr_lr)
            metrics_hr_lr["mse"].append(mse_hr_lr)
            metrics_hr_lr["mae"].append(mae_hr_lr)
            metrics_hr_lr["lsd"].append(lsd_hr_lr)
            metrics_hr_lr["stoi"].append(stoi_hr_lr)
            metrics_hr_lr["pesq"].append(pesq_hr_lr)
        else:
            snr_hr_lr = float("nan")
            lsd_hr_lr = float("nan")

        # SR vs LR metrics
        if lr_path is not None:
            snr_lr_sr = snr_db(wav_sr, wav_lr)
            mse_lr_sr = mse(wav_lr, wav_sr)
            mae_lr_sr = mae(wav_lr, wav_sr)
            lsd_lr_sr = log_spectral_distance(wav_lr, wav_sr)
            stoi_lr_sr, pesq_lr_sr = compute_stoi_pesq(wav_lr, wav_sr, args.sample_rate)

            metrics_lr_sr["snr"].append(snr_lr_sr)
            metrics_lr_sr["mse"].append(mse_lr_sr)
            metrics_lr_sr["mae"].append(mae_lr_sr)
            metrics_lr_sr["lsd"].append(lsd_lr_sr)
            metrics_lr_sr["stoi"].append(stoi_lr_sr)
            metrics_lr_sr["pesq"].append(pesq_lr_sr)
        else:
            snr_lr_sr = float("nan")
            lsd_lr_sr = float("nan")

        print(
            f"{sr_path.name} | "
            f"[HR–SR] SNR={snr_hr_sr:.2f} dB, LSD={lsd_hr_sr:.3f} | "
            f"[HR–LR] SNR={snr_hr_lr:.2f} dB, LSD={lsd_hr_lr:.3f} | "
            f"[LR–SR] SNR={snr_lr_sr:.2f} dB, LSD={lsd_lr_sr:.3f}"
        )

        spec_out = specs_dir / f"{base}_HR-LR-SR.png"
        save_spectrogram_triple(
            wav_hr=wav_hr,
            wav_lr=wav_lr,
            wav_sr=wav_sr,
            sample_rate=args.sample_rate,
            out_path=spec_out,
        )

    if metrics_hr_sr["snr"]:
        print("\n=== AVERAGE METRICS: HR vs SR ===")
        print(f"SNR (dB): {safe_mean(metrics_hr_sr['snr']):.4f}")
        print(f"MSE:      {safe_mean(metrics_hr_sr['mse']):.6f}")
        print(f"MAE:      {safe_mean(metrics_hr_sr['mae']):.6f}")
        print(f"LSD:      {safe_mean(metrics_hr_sr['lsd']):.6f}")
        print(f"STOI:     {safe_mean(metrics_hr_sr['stoi']):.4f}")
        print(f"PESQ:     {safe_mean(metrics_hr_sr['pesq']):.4f}")
    else:
        print("\n[WARN] No valid HR–SR pairs were evaluated.")

    if metrics_hr_lr["snr"]:
        print("\n=== AVERAGE METRICS: HR vs LR ===")
        print(f"SNR (dB): {safe_mean(metrics_hr_lr['snr']):.4f}")
        print(f"MSE:      {safe_mean(metrics_hr_lr['mse']):.6f}")
        print(f"MAE:      {safe_mean(metrics_hr_lr['mae']):.6f}")
        print(f"LSD:      {safe_mean(metrics_hr_lr['lsd']):.6f}")
        print(f"STOI:     {safe_mean(metrics_hr_lr['stoi']):.4f}")
        print(f"PESQ:     {safe_mean(metrics_hr_lr['pesq']):.4f}")
    else:
        print("\n[WARN] No valid HR–LR pairs were evaluated.")

    if metrics_lr_sr["snr"]:
        print("\n=== AVERAGE METRICS: SR vs LR ===")
        print(f"SNR (dB): {safe_mean(metrics_lr_sr['snr']):.4f}")
        print(f"MSE:      {safe_mean(metrics_lr_sr['mse']):.6f}")
        print(f"MAE:      {safe_mean(metrics_lr_sr['mae']):.6f}")
        print(f"LSD:      {safe_mean(metrics_lr_sr['lsd']):.6f}")
        print(f"STOI:     {safe_mean(metrics_lr_sr['stoi']):.4f}")
        print(f"PESQ:     {safe_mean(metrics_lr_sr['pesq']):.4f}")
    else:
        print("\n[WARN] No valid LR–SR pairs were evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SR WAV files against HR & LR, "
            "and save spectrograms (HR/LR/SR)."
        )
    )
    parser.add_argument(
        "--hr_dir",
        type=str,
        default="data/hr",
        help="Directory with HR reference files (e.g., data/hr)",
    )
    parser.add_argument(
        "--low_sr_dir",
        type=str,
        default="data/low_sr",
        help="Directory with LR (8 kHz) files (e.g., data/low_sr)",
    )
    parser.add_argument(
        "--sr_dir",
        type=str,
        default="",
        help="Directory with SR output files (e.g., outputs/). Ignored if --sr_file is set.",
    )
    parser.add_argument(
        "--sr_file",
        type=str,
        default="",
        help="Evaluate a single SR file (e.g., outputs/p225_sr.wav). If set, --sr_dir is ignored.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Sample rate used for evaluation (22050 for your 22k pipeline).",
    )
    args = parser.parse_args()
    main(args)
