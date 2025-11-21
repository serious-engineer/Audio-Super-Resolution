# src/utils/mel_utils.py

import torch
from typing import Tuple

from vocos.feature_extractors import MelSpectrogramFeatures


class LogMelExtractor(torch.nn.Module):
    """
    Mel-spectrogram extractor compatible with Vocos-48k.

    This wraps Vocos' MelSpectrogramFeatures so that the same mel definition is
    used for:
      - Training targets (HR mel)
      - Inputs to the generator (LR mel)
      - Features passed to the Vocos vocoder in inference.

    Output shape (per call):
        input  x: [B, T] or [B, 1, T] or [T]
        output: [B, 1, n_mels, T_frames]
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 256,
        n_mels: int = 128,
        padding: str = "center",
        apply_log: bool = False,
        log_offset: float = 1e-6,
    ):
        super().__init__()

        # This must match the config used to train kittn/vocos-mel-48khz-alpha1
        self.melspec = MelSpectrogramFeatures(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            padding=padding,
        )
        self.apply_log = apply_log
        self.log_offset = log_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram.

        Args:
            x: [T], [B, T], or [B, 1, T] waveform in [-1, 1].

        Returns:
            mel: [B, 1, n_mels, T_frames]
        """
        # Normalize shape to [B, T]
        if x.dim() == 1:
            # [T] -> [1, T]
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:
            # [B, 1, T] -> [B, T]
            x = x[:, 0, :]

        # Now x is [B, T]
        mel = self.melspec(x)  # [B, n_mels, T_frames]

        if self.apply_log:
            mel = torch.log(mel + self.log_offset)

        # Add channel axis to match rest of the pipeline: [B, 1, n_mels, T]
        mel = mel.unsqueeze(1)
        return mel


def split_low_high_mel(
    mel: torch.Tensor,
    split_bin: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split mel into low- and high-frequency parts along the mel axis.

    Args:
        mel: [B, 1, n_mels, T]
        split_bin: index along mel axis that separates low from high.

    Returns:
        low_mel:  [B, 1, split_bin, T]
        high_mel: [B, 1, n_mels - split_bin, T]
    """
    low = mel[:, :, :split_bin, :]
    high = mel[:, :, split_bin:, :]
    return low, high
