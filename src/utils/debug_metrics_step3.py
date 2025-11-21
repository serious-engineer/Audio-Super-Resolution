import torch
import torchaudio

from metrics import snr_db, log_spectral_distance, compute_stoi_pesq  # <-- your file

# CHANGE THIS: point to one clean HR 48kHz file from your dataset
HR_WAV_PATH = "src/utils/p225_001.wav"


def main():
    # -----------------------------------
    # Load HR file
    # -----------------------------------
    hr, sr = torchaudio.load(HR_WAV_PATH)  # hr: [C, T]
    hr = hr.mean(dim=0, keepdim=True)      # make mono [1, T]

    print(f"Loaded HR file at {sr} Hz, shape={hr.shape}")

    # ===================================
    # TEST 1: HR vs HR (sanity check)
    # ===================================
    print("\n===== TEST 1: HR vs HR (same signal) =====")
    snr1 = snr_db(hr, hr)
    lsd1 = log_spectral_distance(hr, hr)
    stoi1, pesq1 = compute_stoi_pesq(hr, hr, sr_ref=sr)

    print(f"SNR  (dB): {snr1}")
    print(f"LSD       : {lsd1}")
    print(f"STOI      : {stoi1}")
    print(f"PESQ      : {pesq1}")

    # What you should see (roughly):
    # - SNR: very large (not negative!)
    # - LSD: very close to 0
    # - STOI: ~1.0
    # - PESQ: close to max (~4+ for clean speech)

    # ===================================
    # TEST 2: Simple LR->HR baseline vs HR
    # ===================================
    print("\n===== TEST 2: Upsampled LR vs HR =====")

    # Example: make LR at 16 kHz (or 8 kHz if you want more aggressive)
    low_sr = 16000

    # Downsample HR -> LR
    lr = torchaudio.functional.resample(hr, sr, low_sr)   # [1, T_low]
    # Upsample LR -> back to original sr
    up = torchaudio.functional.resample(lr, low_sr, sr)   # [1, T_up]

    snr2 = snr_db(hr, up)
    lsd2 = log_spectral_distance(hr, up)
    stoi2, pesq2 = compute_stoi_pesq(hr, up, sr_ref=sr)

    print(f"SNR  (dB): {snr2}")
    print(f"LSD       : {lsd2}")
    print(f"STOI      : {stoi2}")
    print(f"PESQ      : {pesq2}")

    # Rough expectations:
    # - SNR: > 10 dB (definitely NOT negative)
    # - LSD: maybe ~1–3 (not 7–8)
    # - STOI: still fairly high (~0.9+)
    # - PESQ: around 3–4, not like 2.0 if everything is clean


if __name__ == "__main__":
    main()
