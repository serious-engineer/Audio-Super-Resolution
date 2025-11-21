import argparse
import os
from pathlib import Path

import torchaudio


def make_low_sr_dataset(
    hr_dir: str,
    low_sr_dir: str,
    sample_rate: int,
    low_sr: int,
):
    hr_path = Path(hr_dir)
    low_path = Path(low_sr_dir)
    low_path.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in hr_path.iterdir()
        if f.suffix.lower() in [".wav", ".flac", ".ogg"]
    ]
    if not files:
        raise RuntimeError(f"No audio files found in {hr_dir}")

    print(f"Found {len(files)} files in {hr_dir}")
    print(f"Creating low_sr versions in {low_sr_dir} (SR={sample_rate}, low_sr={low_sr})")

    for f in files:
        wav, sr = torchaudio.load(str(f))

        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample to target fullband SR
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        # 48k -> low_sr -> 48k
        wav_low = torchaudio.functional.resample(wav, sample_rate, low_sr)
        wav_low_up = torchaudio.functional.resample(wav_low, low_sr, sample_rate)

        # save with same filename into low_sr_dir
        out_path = low_path / f.name
        out_path = out_path.with_suffix(".wav")  # normalize to wav
        torchaudio.save(str(out_path), wav_low_up, sample_rate)
        print(f"Saved {out_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute low_sr audio files from high-res files."
    )
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root data dir containing hr/ subfolder.")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--low_sr", type=int, default=8000)
    args = parser.parse_args()

    hr_dir = os.path.join(args.data_root, "hr")
    low_dir = os.path.join(args.data_root, "low_sr")

    make_low_sr_dataset(hr_dir, low_dir, args.sample_rate, args.low_sr)


if __name__ == "__main__":
    main()
