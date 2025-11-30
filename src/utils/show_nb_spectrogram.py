#!/usr/bin/env python3
import sys
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python show_spectrogram_48k.py path/to/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    target_sr = 96000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    n_fft = 2048
    hop_length = n_fft // 4
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    S_mag = np.abs(S) + 1e-10
    S_db = 20 * np.log10(S_mag / np.max(S_mag))

    freqs = np.linspace(0, sr / 2, S_db.shape[0])
    times = np.arange(S_db.shape[1]) * hop_length / sr

    plt.figure(figsize=(10, 4))
    plt.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram (0–12 kHz)")
    plt.ylim(0, 11000)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
