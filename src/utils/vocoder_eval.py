# src/utils/vocoder_eval.py
import argparse
from pathlib import Path
import torch
import torchaudio
from src.utils.vocos_22k import Vocos22kVocoder
from src.utils.mel_utils import LogMelExtractor
from src.utils.metrics import (
    snr_db,
    mse,
    mae,
    log_spectral_distance,
    compute_stoi_pesq,
)

def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav

def evaluate_vocoder(
    hr_dir: str,
    sample_rate: int = 22050,
    device: str = "cpu",
    max_files: int = 0,
    save_recon: bool = False,
    out_dir: str = "vocoder22_recon",
):
    device = torch.device(device)
    hr_dir = Path(hr_dir)

    print(f"\n[Loading Vocos-22k model on {device}]")
    vocoder = Vocos22kVocoder(device=device)
    vocoder.eval()
    mel_extractor = LogMelExtractor(
        sample_rate=sample_rate,
        n_mels=80,
        apply_log=False,
    ).to(device)
    wav_list = sorted(list(hr_dir.glob("*.wav")))
    if len(wav_list) == 0:
        print(f"[ERROR] No .wav files found in {hr_dir}")
        return

    if max_files > 0:
        wav_list = wav_list[:max_files]

    print(f"\n[Evaluating Vocos-22k on {len(wav_list)} HR files]\n")

    if save_recon:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Saving reconstructions to: {out_dir}]\n")

    for path in wav_list:
        path_str = str(path)
        wav_hr = load_audio(path_str, sample_rate=sample_rate).to(device)

        with torch.no_grad():
            mel = mel_extractor(wav_hr)
            mel = mel.squeeze(1)
            wav_voc = vocoder(mel)

        min_len = min(wav_hr.shape[-1], wav_voc.shape[-1])
        ref = wav_hr[..., :min_len]
        rec = wav_voc[..., :min_len]

        ref_1d = ref.squeeze().cpu()
        rec_1d = rec.squeeze().cpu()

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

        if save_recon:
            out_path = Path(out_dir) / f"{path.stem}_vocos22.wav"
            torchaudio.save(str(out_path), rec_1d.unsqueeze(0), sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Vocos-22k vocoder on HR wav files."
    )
    parser.add_argument(
        "--hr_dir",
        type=str,
        required=True,
        help="Directory containing 22 kHz HR .wav files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Must be 22050 for vocos-22khz.",
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
        help="Use CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--save_recon",
        action="store_true",
        help="If set, save reconstructed wavs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="vocoder22_recon",
        help="Output directory for reconstructed wavs.",
    )

    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_vocoder(
        hr_dir=args.hr_dir,
        sample_rate=args.sample_rate,
        device=device,
        max_files=args.max_files,
        save_recon=args.save_recon,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()