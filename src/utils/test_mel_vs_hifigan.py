import torch
import torchaudio
import torchaudio.transforms as T

from src.utils.hifigan_48k import HiFiGAN48k
from src.utils.mel_utils import LogMelExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------
# 1. LOAD A CLEAN HR WAV
# --------------------------------------
wav, sr = torchaudio.load("data/hr/p225_001.wav")
if wav.size(0) > 1:
    wav = wav.mean(dim=0, keepdim=True)
wav = torchaudio.functional.resample(wav, sr, 48000).to(device)

# --------------------------------------
# 2. USE YOUR MEL EXTRACTOR
# --------------------------------------
my_mel = LogMelExtractor(
    sample_rate=48000,
    n_fft=2048,
    hop_length=256,
    n_mels=128,
    apply_log=True,
).to(device)

mel = my_mel(wav)      # [1, 1, 128, T]
mel = mel.squeeze(1)   # [1, 128, T] ← what HiFiGAN expects

# --------------------------------------
# 3. RUN HIFIGAN ON IT
# --------------------------------------
vocoder = HiFiGAN48k(device=device).eval()

with torch.no_grad():
    wav_out = vocoder(mel)   # [1, T]

torchaudio.save("test_my_mel_hifigan.wav", wav_out.cpu(), 48000)
print("Saved: test_my_mel_hifigan.wav")
