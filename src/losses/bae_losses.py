# src/losses/bae_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectralConvergence(nn.Module):
    """
    Spectral convergence in mel-domain.
    This is similar to STFT-SC but applied directly on mel spectrograms.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, M_fake, M_real):
        # Assume shapes [B, 1, n_mels, T] or [B, n_mels, T]
        if M_fake.dim() == 4:
            M_fake = M_fake.squeeze(1)
            M_real = M_real.squeeze(1)

        diff = torch.norm(M_real - M_fake, p='fro')
        denom = torch.norm(M_real, p='fro') + self.eps
        return diff / denom


class MelLoss(nn.Module):
    """
    A stable and perceptual mel-domain loss:

    L = λ1 * L1
      + λ2 * L2
      + λ3 * Mel-Spectral-Convergence

    Works safely on mel spectrograms (unlike waveform STFT).
    """
    def __init__(self, lambda_l1=1.0, lambda_l2=1.0, lambda_sc=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_sc = lambda_sc

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.sc = MelSpectralConvergence()

    def forward(self, M_fake, M_real):
        if M_fake.dim() == 4:
            M_fake = M_fake.squeeze(1)
            M_real = M_real.squeeze(1)

        l1_loss = self.l1(M_fake, M_real)
        l2_loss = self.l2(M_fake, M_real)
        sc_loss = self.sc(M_fake, M_real)

        total = (
            self.lambda_l1 * l1_loss +
            self.lambda_l2 * l2_loss +
            self.lambda_sc * sc_loss
        )

        stats = {
            "mel_l1": l1_loss.item(),
            "mel_mse": l2_loss.item(),
            "mel_sc": sc_loss.item(),
            "total": total.item(),
            "stft_total": total.item(),  # <- add this line
        }

        return total, stats
