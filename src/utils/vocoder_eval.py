# src/utils/vocoder_eval.py

import argparse
from pathlib import Path

import torch
import torchaudio

# Vocos
from vocos import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures

# HiFi-GAN 48k
from src.utils.hifigan_48k import HiFiGAN48k

from src.utils.metrics import (
    snr_db,
    mse,
    mae,
    log_spectral_distance,
    compute_stoi_pesq,
)

def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    """
    Load audio as mono, resample to sample_rate if needed.
    Returns [1, T] tensor.
    """
    wav, sr = torchaudio.load(path)

    # Mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    return wav


def evaluate_vocoder(
    hr_dir: str,
    sample_rate: int = 48000,
    device: str = "cpu",
    vocoder_type: str = "vocos48",
    max_files: int = 0,
    save_recon: bool = False,
    out_dir: str = "vocoder_recon",
):
    """
    Evaluate a vocoder (Vocos-48k or HiFi-GAN 48k) by reconstructing HR
    files from their own mel spectrograms and comparing to the original.
    """
    device = torch.device(device)
    hr_dir = Path(hr_dir)

    # ---------------------------------------------------------
    # 1. Load vocoder + mel extractor
    # ---------------------------------------------------------
    if vocoder_type == "vocos48":
        print(f"\n[Loading Vocos 48 kHz model on {device}]")
        vocoder = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
        vocoder.eval()

        mel_extractor = MelSpectrogramFeatures(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=256,
            n_mels=128,
            padding="center",
        ).to(device)

    elif vocoder_type == "hifigan48":
        print(f"\n[Loading HiFi-GAN 48 kHz model on {device}]")
        vocoder = HiFiGAN48k(device=device)
        mel_extractor = HiFiGANMelExtractor(device=device)

    else:
        raise ValueError("vocoder_type must be one of: 'vocos48', 'hifigan48'")

    # ---------------------------------------------------------
    # 2. Collect HR wav files
    # ---------------------------------------------------------
    wav_list = sorted(list(hr_dir.glob("*.wav")))
    if len(wav_list) == 0:
        print(f"[ERROR] No .wav files found in {hr_dir}")
        return

    if max_files > 0:
        wav_list = wav_list[:max_files]

    print(f"\n[Evaluating {vocoder_type} on {len(wav_list)} clean HR files]\n")

    if save_recon:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Info] Reconstructed WAVs will be saved to: {out_dir}\n")

    # ---------------------------------------------------------
    # 3. Loop over files
    # ---------------------------------------------------------
    for path in wav_list:
        path_str = str(path)

        # 3.1 Load HR waveform [1, T]
        wav_hr = load_audio(path_str, sample_rate=sample_rate).to(device)

        with torch.no_grad():
            # 3.2 Compute mel
            mel = mel_extractor(wav_hr)  # e.g. [1, n_mels, T_frames]

            # Normalize dims to [B, n_mels, T]
            if mel.dim() == 4:
                mel = mel.squeeze(1)  # [B, n_mels, T]
            elif mel.dim() == 2:
                mel = mel.unsqueeze(0)  # [1, n_mels, T]

            # 3.3 Decode
            wav_voc = vocoder.decode(mel)  # expected [1, T] or [1, 1, T]

        # 3.4 Align lengths and squeeze to 1D for metrics
        # wav_hr: [1, T_hr], wav_voc: [1, T_voc] or [1, 1, T_voc]
        if wav_voc.dim() == 3:
            # [B, 1, T] -> [B, T]
            wav_voc = wav_voc.squeeze(1)

        min_len = min(wav_hr.shape[-1], wav_voc.shape[-1])
        ref = wav_hr[..., :min_len]  # [1, T]
        rec = wav_voc[..., :min_len]  # [1, T]

        # Squeeze to 1D for STOI/PESQ and other metrics
        ref_1d = ref.squeeze().detach().cpu()  # [T]
        rec_1d = rec.squeeze().detach().cpu()  # [T]

        # 3.5 Metrics
        snr_val = snr_db(ref_1d, rec_1d)
        mse_val = mse(ref_1d, rec_1d)
        mae_val = mae(ref_1d, rec_1d)
        lsd_val = log_spectral_distance(ref_1d, rec_1d)

        stoi_val, pesq_val = compute_stoi_pesq(ref_1d, rec_1d, sample_rate)

        stoi_str = "N/A" if stoi_val is None else f"{stoi_val:.4f}"
        pesq_str = "N/A" if pesq_val is None else f"{pesq_val:.4f}"

        print(f"{path.name}:")
        print(f"  SNR:   {snr_val:.2f} dB")
        print(f"  MSE:   {mse_val:.6e}")
        print(f"  MAE:   {mae_val:.6e}")
        print(f"  LSD:   {lsd_val:.4f}")
        print(f"  STOI:  {stoi_str}")
        print(f"  PESQ:  {pesq_str}\n")

        # 3.6 Optional: save reconstruction
        if save_recon:
            out_path = Path(out_dir) / f"{path.stem}_{vocoder_type}.wav"
            wav_out = rec_1d.unsqueeze(0)  # [1, T] for torchaudio.save
            torchaudio.save(str(out_path), wav_out, sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vocoder quality on HR wav files."
    )
    parser.add_argument(
        "--hr_dir",
        type=str,
        required=True,
        help="Directory containing 48 kHz HR .wav files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="Sample rate to use (48 kHz).",
    )
    parser.add_argument(
        "--vocoder",
        type=str,
        required=True,
        choices=["vocos48", "hifigan48"],
        help="Which vocoder to evaluate.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If >0, evaluate only the first N files.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--save_recon",
        action="store_true",
        help="If set, save reconstructed wavs to --out_dir.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="vocoder_recon",
        help="Output directory for reconstructed wavs (if --save_recon).",
    )

    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_vocoder(
        hr_dir=args.hr_dir,
        sample_rate=args.sample_rate,
        device=device,
        vocoder_type=args.vocoder,
        max_files=args.max_files,
        save_recon=args.save_recon,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
