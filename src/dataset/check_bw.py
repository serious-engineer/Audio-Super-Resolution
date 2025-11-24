import argparse
import torchaudio
import torch
import numpy as np

def check_bandlimit(path, cutoff_hz=4000):
    print(f"\n[Checking] {path}")
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)

    n_fft = 4096
    hop = 1024

    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        return_complex=True
    )[0]

    mag = spec.abs()
    power = mag**2
    freqs = torch.linspace(0, sr/2, n_fft//2 + 1)
    below_mask = freqs <= cutoff_hz
    above_mask = freqs > cutoff_hz

    power_below = power[below_mask].sum().item()
    power_above = power[above_mask].sum().item()

    ratio = power_above / (power_below + 1e-12)

    print(f"Sample Rate     : {sr} Hz")
    print(f"Cutoff Frequency: {cutoff_hz} Hz\n")
    print(f"Power < {cutoff_hz} Hz : {power_below:.4e}")
    print(f"Power > {cutoff_hz} Hz : {power_above:.4e}")
    print(f"Ratio (above/below): {ratio:.4e}")

    if ratio < 1e-3:
        print(f"\n=> ✔ YES: Signal is effectively bandlimited below {cutoff_hz/1000:.1f} kHz")
    else:
        print(f"\n=> ✘ NO: Signal has significant energy above {cutoff_hz/1000:.1f} kHz")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Check if audio is bandlimited")
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--cutoff", type=float, default=4000,
                        help="Cutoff frequency in Hz (default: 4000)")
    args = parser.parse_args()

    check_bandlimit(args.input, args.cutoff)


if __name__ == "__main__":
    main()
