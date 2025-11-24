# src/utils/vocos_22k.py
import torch
import torch.nn as nn
from vocos import Vocos

class Vocos22kVocoder(nn.Module):
    """
    Wrapper around the pretrained BSC-LT/vocos-mel-22khz model.
    """
    def __init__(self, device=None):
        super().__init__()
        try:
            self.vocos = Vocos.from_pretrained("BSC-LT/vocos-mel-22khz")
        except Exception as e:
            raise RuntimeError(
                "Failed to load Vocos 22 kHz vocoder.\n"
                "Make sure you installed the correct Vocos version:\n"
                "  pip install git+https://github.com/langtech-bsc/vocos.git@matcha"
            ) from e
        if device is not None:
            self.vocos = self.vocos.to(device)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        mel = mel.float()
        audio = self.vocos.decode(mel)
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio[:, 0, :]
        return audio
