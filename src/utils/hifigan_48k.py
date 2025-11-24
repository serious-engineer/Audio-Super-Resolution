# src/vocoders/hifigan_48k.py
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn

class HiFiGAN48k(nn.Module):
    """
    Wrapper for hifi-gan 48k mel vocoder.
    """
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model = torch.hub.load(
            "vtuber-plan/hifi-gan:v0.3.0",
            "hifigan_48k",
            force_reload=False,
        ).to(self.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0) 

        audio = self.model(mel.to(self.device)) 

        if audio.dim() == 3:
            audio = audio.squeeze(1) 

        return audio

    @torch.no_grad()
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        return self.forward(mel)

class HiFiGANMelExtractor(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device)

        self.n_fft = 2048
        self.hop = 512
        self.win = 2048
        self.n_mels = 128
        self.sr = 48000

        pad = int((self.n_fft - self.hop) / 2)

        self.spec = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win,
            hop_length=self.hop,
            pad=pad,
            power=None,
            center=False,
        ).to(self.device)

        self.mel_scale = T.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sr,
            n_stft=self.n_fft // 2 + 1,
        ).to(self.device)

    @torch.no_grad()
    def forward(self, wav):
        spec = self.spec(wav)
        mag = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-6)
        mel = self.mel_scale(mag)
        return mel
