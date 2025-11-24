#!/usr/bin/env python3
"""
audio_properties.py

Usage:
    python audio_properties.py path/to/audio.wav

Prints:
    - sampling rate
    - number of channels
    - duration (seconds)
    - total samples
    - bit depth (if available)
    - dtype
    - file format
    - min / max amplitude
    - RMS
"""

import argparse
import soundfile as sf
import numpy as np
import librosa

def get_audio_properties(path):
    info = sf.info(path)
    audio, sr = sf.read(path)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio_mono = audio.mean(axis=1)
    else:
        audio_mono = audio

    duration = len(audio_mono) / sr
    rms = np.sqrt(np.mean(audio_mono**2))
    peak = np.max(np.abs(audio_mono))

    props = {
        "File":               path,
        "Format":             info.format,
        "Subtype":            info.subtype,
        "Bit Depth":          info.subtype_info,
        "Sample Rate":        sr,
        "Channels":           info.channels,
        "Total Samples":      len(audio),
        "Duration (seconds)": duration,
        "Data Type":          audio.dtype,
        "Min Amplitude":      float(audio_mono.min()),
        "Max Amplitude":      float(audio_mono.max()),
        "RMS":                float(rms),
        "Peak Level":         float(peak),
    }

    return props


def main():
    parser = argparse.ArgumentParser(description="Print properties of an audio file.")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    args = parser.parse_args()

    props = get_audio_properties(args.audio_path)

    print("\n=== AUDIO FILE PROPERTIES ===")
    for k, v in props.items():
        print(f"{k:20s}: {v}")

    print("==============================\n")


if __name__ == "__main__":
    main()
