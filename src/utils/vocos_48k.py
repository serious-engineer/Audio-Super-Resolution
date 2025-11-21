import torch
import torch.nn as nn
from vocos import Vocos


class Vocos48kVocoder(nn.Module):
    """
    Wrapper around the pretrained kittn/vocos-mel-48khz-alpha1 model.

    Expects mel with shape [B, n_mels, T] where:
      - sample_rate = 48000
      - n_fft       = 2048
      - hop_length  = 256
      - n_mels      = 128
      - padding     = "center"

    These must match the LogMelExtractor configuration.
    """

    def __init__(self, device=None):
        super().__init__()

        try:
            self.vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1")
        except Exception as e:
            raise RuntimeError(
                "Failed to load Vocos 48 kHz vocoder. "
                "Make sure 'vocos' is installed: pip install vocos"
            ) from e

        if device is not None:
            self.vocos = self.vocos.to(device)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T] log-mel (same definition as used in training)

        Returns:
            audio: [B, T_samples] waveform at 48 kHz.
        """
        if mel.dim() == 4:
            # [B, 1, n_mels, T] → [B, n_mels, T]
            mel = mel.squeeze(1)

        mel = mel.float()
        audio = self.vocos.decode(mel)  # [B, T] or [B, 1, T]

        # Ensure shape [B, T]
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio[:, 0, :]

        return audio
