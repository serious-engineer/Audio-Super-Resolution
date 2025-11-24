# src/infer.py
import argparse
import os
import torch
import torchaudio
from pathlib import Path
from src.models.audio_srgan import AudioSRGenerator
from src.utils.mel_utils import LogMelExtractor
from src.utils.simple_vocoder import MelGriffinLimVocoder
from src.utils.vocos_48k import Vocos48kVocoder
from src.utils.hifigan_48k import HiFiGAN48k
from src.utils.vocos_22k import Vocos22kVocoder
from src.utils.metrics import *

def load_audio(path: str, sr: int) -> torch.Tensor:
    # Load mono audio and resample to sr if needed.
    wav, s = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav

def find_hr_match(low_sr_path: str, hr_dir: str):
    base = Path(low_sr_path).stem
    hr_dir = Path(hr_dir)
    for ext in [".wav", ".flac", ".ogg"]:
        cand = hr_dir / f"{base}{ext}"
        if cand.is_file():
            return str(cand)
    return None

def run_single_inference(
    low_path,
    G,
    log_mel_extractor,
    vocoder,
    sample_rate,
    hr_dir,
    output_dir,
    device,
):
    low_path = Path(low_path)
    base = low_path.stem

    #Load low-res waveform
    wav_low = load_audio(str(low_path), sample_rate).to(device)  # [1, T]

    #Mel to Generator to Mel_SR
    with torch.no_grad():
        mel_lr = log_mel_extractor(wav_low)  # [1, 1, n_mels, Tm]
        mel_sr = G(mel_lr)

        # For vocoders, we need [B, n_mels, T]
        mel_sr_for_vocoder = mel_sr.squeeze(1)  # [1, n_mels, T]

        #  Vocoder
        wav_sr = vocoder(mel_sr_for_vocoder) 
        if wav_sr.dim() == 1:
            wav_sr = wav_sr.unsqueeze(0)

    # Save SR waveform
    out_path = Path(output_dir) / f"{base}_sr.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav_sr.cpu(), sample_rate)
    print(f"[Saved] {out_path}")

    # If HR exists compute metrics
    if hr_dir:
        hr_path = find_hr_match(str(low_path), hr_dir)
        if hr_path is None:
            print(f"[WARN] No HR match found for {base}")
            return None

        wav_hr = load_audio(hr_path, sample_rate).to(device)

        ref = wav_hr.mean(dim=0, keepdim=True).cpu()
        est = wav_sr.mean(dim=0, keepdim=True).cpu()  
        # SNR / SEG SNR / LSD
        snr_val = snr_db(ref, est, use_alignment=True)
        seg_snr_val = seg_snr_db(ref, est)
        lsd_val = log_spectral_distance(ref, est)

        # MSE / MAE
        mse_val = mse(ref, est)
        mae_val = mae(ref, est)

        # STOI / PESQ
        stoi_val, pesq_val = compute_stoi_pesq(ref, est, sr_ref=sample_rate)

        print(
            f"[Metrics] {base}: "
            f"SNR={snr_val:.2f} dB, "
            f"SEG SNR={seg_snr_val:.2f} dB, "
            f"LSD={lsd_val:.3f}, "
            f"STOI={stoi_val}, PESQ={pesq_val}"
        )

        return {
            "file": base,
            "snr": snr_val,
            "mse": mse_val,
            "mae": mae_val,
            "lsd": lsd_val,
            "stoi": stoi_val,
            "pesq": pesq_val,
        }

    return None

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    # Load generator
    G = AudioSRGenerator(num_res_blocks=args.num_res_blocks).to(device)
    ckpt = torch.load(args.generator_ckpt, map_location=device)

    # ckpt_last.pt (full checkpoint)
    if isinstance(ckpt, dict) and "G" in ckpt:
        print("[INFO] Loading weights from ckpt_last.pt (G key)")
        G.load_state_dict(ckpt["G"])
    # generator_final.pt (pure state_dict)
    elif isinstance(ckpt, dict):
        print("[INFO] Loading pure state_dict (generator_final.pt)")
        G.load_state_dict(ckpt)
    else:
        raise RuntimeError("Unexpected checkpoint format")

    G.eval()

    # Log-mel extractor (Vocos-22k compatible)
    log_mel = LogMelExtractor(
        sample_rate=args.sample_rate,  # 22050
        n_mels=args.n_mels,           # 80
        # n_fft/hop/padding are fixed inside to 1024/256/"same"
    ).to(device)


    # Choose vocoder
    if args.vocoder == "griffin":
        vocoder = MelGriffinLimVocoder(
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
        ).to(device)
    elif args.vocoder == "vocos48":
        if args.sample_rate != 48000:
            print(
                f"[WARN] Vocos 48k vocoder is trained for 48000 Hz, "
                f"but sample_rate={args.sample_rate}"
            )
        vocoder = Vocos48kVocoder(device=device)
    elif args.vocoder == "vocos22":
        if args.sample_rate != 22050:
            print(
                f"[WARN] Vocos 22k vocoder is trained for 22050 Hz, "
                f"but sample_rate={args.sample_rate}"
            )
        if args.n_mels != 80:
            print(
                f"[WARN] Vocos 22k vocoder expects n_mels=80, "
                f"but n_mels={args.n_mels}"
            )
        vocoder = Vocos22kVocoder(device=device)
    elif args.vocoder == "hifigan48":
        vocoder = HiFiGAN48k(device=device)
    else:
        raise ValueError(f"Unknown vocoder type: {args.vocoder}")

    vocoder.eval()
    print(f"[Vocoder] Using {args.vocoder} vocoder")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_wav:
        run_single_inference(
            low_path=args.input_wav,
            G=G,
            log_mel_extractor=log_mel,
            vocoder=vocoder,
            sample_rate=args.sample_rate,
            hr_dir=args.hr_dir,
            output_dir=args.output_dir,
            device=device,
        )
        return

    low_dir = Path(args.low_sr_dir)
    files = sorted(
        f for f in low_dir.iterdir()
        if f.suffix.lower() in [".wav", ".flac", ".ogg"]
    )

    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"\n[Batch Inference] {len(files)} files\n")

    for f in files:
        run_single_inference(
            low_path=f,
            G=G,
            log_mel_extractor=log_mel,
            vocoder=vocoder,
            sample_rate=args.sample_rate,
            hr_dir=args.hr_dir,
            output_dir=args.output_dir,
            device=device,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_wav",
        type=str,
        default="",
        help="Run on single low_sr file. If empty → batch mode.",
    )
    parser.add_argument(
        "--low_sr_dir",
        type=str,
        default="data/low_sr",
        help="Directory containing low_sr files.",
    )
    parser.add_argument(
        "--hr_dir",
        type=str,
        default="data/hr",
        help="Directory of HR files for metric comparison.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_gan",
        help="Where to save SR outputs.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If >0, limit batch inference to N files.",
    )
    parser.add_argument("--generator_ckpt", type=str, default="checkpoints_gan/generator_final.pt")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--num_res_blocks", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--vocoder",
        type=str,
        default="vocos22",
        choices=["griffin", "vocos22"],
        help="Which vocoder to use for waveform reconstruction.",
    )

    args = parser.parse_args()
    main(args)
