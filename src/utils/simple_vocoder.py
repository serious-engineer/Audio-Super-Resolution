import torch
import torch.nn as nn
import torchaudio


class MelGriffinLimVocoder(nn.Module):
    """
    Very simple vocoder that tries to invert log-mel spectrograms back to a waveform
    using InverseMelScale + Griffin-Lim.
    """
    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 960,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
        )

        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1.0,
            n_iter=32,
        )

    @torch.no_grad()
    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        mel_power = torch.exp(log_mel)
        mel_mag = mel_power.clamp_min(1e-10).sqrt()
        spec_mag = self.inv_mel(mel_mag)
        wav = self.griffin(spec_mag)
        wav = torch.clamp(wav, -1.0, 1.0)
        return wav
