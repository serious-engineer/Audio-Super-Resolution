import argparse
import os
import torchaudio
from tqdm import tqdm
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".ogg")

def resample_and_mono(wav, sr, target_sr):
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav, target_sr

def normalize_audio(wav):
    max_val = wav.abs().max()
    if max_val > 0:
        wav = wav / max_val
    return wav


def prepare_pair_dataset(
    in_root: str,
    out_root: str | None = None,
    hr_sr: int = 22050,
    low_sr: int = 8000,
):
    in_root = Path(in_root)
    if out_root is None:
        out_root = in_root
    else:
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

    hr_dir = out_root / "hr"
    low_dir = out_root / "low_sr"
    hr_dir.mkdir(parents=True, exist_ok=True)
    low_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in sorted(in_root.iterdir())
        if f.is_file()
        and f.suffix.lower() in AUDIO_EXTS
    ]

    if not files:
        raise RuntimeError(f"No audio files with extensions {AUDIO_EXTS} found in {in_root}")

    print(f"Found {len(files)} audio files in {in_root}")
    print(f"HR folder:    {hr_dir}  (sr = {hr_sr} Hz)")
    print(f"low_sr folder:{low_dir} (sr = {low_sr} Hz)")

    for f in tqdm(files, desc="Processing files"):
        wav, sr = torchaudio.load(str(f))

        wav_hr, _ = resample_and_mono(wav, sr, hr_sr)
        wav_hr = normalize_audio(wav_hr)

        base_name = f.stem + ".wav"
        hr_out = hr_dir / base_name
        torchaudio.save(str(hr_out), wav_hr, hr_sr)

        wav_low = torchaudio.functional.resample(wav_hr, hr_sr, low_sr)
        wav_low = normalize_audio(wav_low)
        low_out = low_dir / base_name
        torchaudio.save(str(low_out), wav_low, low_sr)

    print("Done.")
    print(f"HR  folder: {hr_dir}")
    print(f"LR  folder: {low_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a pair of folders (hr/ and low_sr/) from an input folder of audio files.\n"
            "By default: hr at 22050 Hz, low_sr at 8000 Hz."
        )
    )
    parser.add_argument(
        "--in_root",
        type=str,
        required=True,
        help="Input folder containing raw audio files.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data",
        help="Output root folder. If not set, hr/ and low_sr/ are created inside in_root.",
    )
    parser.add_argument(
        "--hr_sr",
        type=int,
        default=22050,
        help="Target sample rate for hr folder (default: 22050).",
    )
    parser.add_argument(
        "--low_sr",
        type=int,
        default=8000,
        help="Target sample rate for low_sr folder (default: 8000).",
    )

    args = parser.parse_args()
    prepare_pair_dataset(
        in_root=args.in_root,
        out_root=args.out_root,
        hr_sr=args.hr_sr,
        low_sr=args.low_sr,
    )

if __name__ == "__main__":
    main()
