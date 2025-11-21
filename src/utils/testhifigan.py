import torch
import torchaudio


from src.utils.hifigan_48k import HifiGAN48Vocoder
from src.utils.mel_hifigan import MelExtractorHifiGAN


HR_WAV_PATH = "src/utils/p225_001.wav"  # full-band HR file

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    wav, sr = torchaudio.load(HR_WAV_PATH)  # [C, T]
    wav = wav.mean(dim=0, keepdim=True)     # mono [1, T]
    wav = wav.to(device)

    mel_extractor = MelExtractorHifiGAN(
        sample_rate=sr,
        n_mels=128,
        device=device
    )

    mel = mel_extractor(wav)  # [B, 1, n_mels, T]
    vocoder = HifiGAN48Vocoder(device=device)
    wav_hat = vocoder(mel)   # [B, T] = [1, T]

    # Ensure it's 2D: [channels, samples]
    wav_to_save = wav_hat.detach().cpu()  # [1, T]

    torchaudio.save(
        "test_hifigan_on_hr.wav",
        wav_to_save,
        sr,
    )


    print("[Saved] test_hifigan_on_hr.wav")

if __name__ == "__main__":
    main()
