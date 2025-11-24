import argparse
import torchaudio
import torchaudio.functional as F
from pathlib import Path

FIXED_OUT_DIR = Path(r"data\low_8k")

def resample_to_8k(input_path):

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    wav, sr = torchaudio.load(str(input_path))
    if sr != 8000:
        print(f"[Resample] {input_path.name}: {sr} Hz → 8000 Hz")
        wav_8k = F.resample(wav, orig_freq=sr, new_freq=8000)
    else:
        print(f"[Info] {input_path.name} is already 8 kHz, copying.")
        wav_8k = wav

    FIXED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIXED_OUT_DIR / input_path.name

    torchaudio.save(str(output_path), wav_8k, 8000)
    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Resample audio to 8 kHz")
    parser.add_argument("--input", required=True, help="Path to audio file")
    args = parser.parse_args()

    resample_to_8k(args.input)


if __name__ == "__main__":
    main()
