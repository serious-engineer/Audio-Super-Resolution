import torch
import torchaudio
from metrics import snr_db, log_spectral_distance, compute_stoi_pesq

HR_WAV_PATH = "src/utils/p225_001.wav"

def main():
    hr, sr = torchaudio.load(HR_WAV_PATH) # [channels, time]
    hr = hr.mean(dim=0, keepdim=True)      

    print(f"Loaded HR file at {sr} Hz, shape={hr.shape}")
    print("\n===== TEST 1: HR vs HR (same signal) =====")
    snr1 = snr_db(hr, hr)
    lsd1 = log_spectral_distance(hr, hr)
    stoi1, pesq1 = compute_stoi_pesq(hr, hr, sr_ref=sr)

    print(f"SNR  (dB): {snr1}")
    print(f"LSD       : {lsd1}")
    print(f"STOI      : {stoi1}")
    print(f"PESQ      : {pesq1}")


    print("\n===== TEST 2: Upsampled LR vs HR =====")
    low_sr = 16000

    # Downsample HR to LR
    lr = torchaudio.functional.resample(hr, sr, low_sr)
    # Upsample LR to back to original sr
    up = torchaudio.functional.resample(lr, low_sr, sr)

    snr2 = snr_db(hr, up)
    lsd2 = log_spectral_distance(hr, up)
    stoi2, pesq2 = compute_stoi_pesq(hr, up, sr_ref=sr)

    print(f"SNR  (dB): {snr2}")
    print(f"LSD       : {lsd2}")
    print(f"STOI      : {stoi2}")
    print(f"PESQ      : {pesq2}")

if __name__ == "__main__":
    main()
