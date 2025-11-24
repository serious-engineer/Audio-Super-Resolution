import torch
import torch.nn as nn
from vocos import Vocos

class Vocos48kVocoder(nn.Module):
    """
    Wrapper around the pretrained vocos-mel-48khz-alpha1 model.
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
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        mel = mel.float()
        audio = self.vocos.decode(mel)
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio[:, 0, :]

        return audio
