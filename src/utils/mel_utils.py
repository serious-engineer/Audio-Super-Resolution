# src/utils/mel_utils.py
import torch
from typing import Tuple
from vocos.feature_extractors import MelSpectrogramFeatures


class LogMelExtractor(torch.nn.Module):
    """
    Mel-spectrogram extractor compatible with vocos-mel-22khz. 
    Produces 80-bin mel-spectrograms from 22.05 kHz audio.
    """
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        apply_log: bool = False,
        log_offset: float = 1e-6,
    ):
        super().__init__()
        if sample_rate != 22050:
            raise ValueError(
                f"LogMelExtractor must use sample_rate=22050 for vocos-mel-22khz, got {sample_rate}"
            )
        if n_mels != 80:
            raise ValueError(
                f"LogMelExtractor must use n_mels=80 for vocos-mel-22khz, got {n_mels}"
            )
        self.melspec = MelSpectrogramFeatures(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            padding="same",
        )
        self.apply_log = apply_log
        self.log_offset = log_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        mel = self.melspec(x)
        if self.apply_log:
            mel = torch.log(mel + self.log_offset)
        mel = mel.unsqueeze(1)
        return mel


def split_low_high_mel(
    mel: torch.Tensor,
    split_bin: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    low = mel[:, :, :split_bin, :]
    high = mel[:, :, split_bin:, :]
    return low, high
