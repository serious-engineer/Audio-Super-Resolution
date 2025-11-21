import torch
import torch.nn as nn
import torchaudio


class MelExtractorHifiGAN(nn.Module):
    """
    Mel extractor compatible with vtuber-plan/hifi-gan 'hifigan_48k'.
    """
    def __init__(self, sample_rate=48000, n_mels=128, device="cpu"):
        super().__init__()
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2,
            center=False,  # IMPORTANT
            power=1.0,
            norm=None,
            mel_scale="htk",
        ).to(device)

    def forward(self, wav):
        """
        wav: [B, T]
        return: log-mel [B, n_mels, T_mel]
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)

        mel = self.mel(wav)  # linear mel

        # apply log clamp EXACTLY like HiFi-GAN
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel
