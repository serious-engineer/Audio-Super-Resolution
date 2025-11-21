import torch
import torch.nn as nn
import torchaudio


class MelGriffinLimVocoder(nn.Module):
    """
    Very simple vocoder that tries to invert log-mel spectrograms back to a waveform
    using InverseMelScale + Griffin-Lim.

    This matches the LogMelExtractor we already use:
      - mel is log(power)
      - power = exp(log_mel)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 480,   # 10 ms at 48 kHz
        win_length: int = 960,   # 20 ms at 48 kHz
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        # Inverse mel: mel [B, n_mels, T] -> linear magnitude [B, n_freq, T]
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
        )

        # Griffin-Lim to reconstruct phase and waveform
        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1.0,          # we pass magnitude, not power
            n_iter=32,
        )

    @torch.no_grad()
    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_mel: [B, n_mels, T]  (log power mel)

        Returns:
            waveform: [B, T_samples] in [-1, 1]
        """
        # Undo log: log(power) -> power
        mel_power = torch.exp(log_mel)

        # Convert power to magnitude
        mel_mag = mel_power.clamp_min(1e-10).sqrt()

        # Invert mel to linear magnitude spectrogram
        spec_mag = self.inv_mel(mel_mag)

        # Griffin-Lim expects [B, n_freq, T] magnitude
        wav = self.griffin(spec_mag)

        # Clamp just in case
        wav = torch.clamp(wav, -1.0, 1.0)
        return wav
