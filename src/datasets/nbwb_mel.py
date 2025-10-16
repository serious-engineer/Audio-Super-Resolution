import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import soundfile as sf
import librosa
import numpy as np

def wav2mel(y, sr, n_fft, hop, win, n_mels, fmin, fmax):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    S = np.log10(1e-6 + S).astype(np.float32)
    return S  # shape [n_mels, T]

class NBWBMelPairs(data.Dataset):
    def __init__(self, csv_path, sr=16000, n_fft=512, hop=128, win=512, n_mels=80, fmin=20, fmax=8000, seg_seconds=1.0):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.n_mels, self.fmin, self.fmax = n_mels, fmin, fmax
        self.seg_samples = int(seg_seconds * sr) if seg_seconds else None

    def __len__(self):
        return len(self.df)

    def _load_center_crop(self, path):
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        if self.seg_samples:
            if len(y) < self.seg_samples:
                pad = self.seg_samples - len(y)
                y = np.pad(y, (0, pad))
            start = max(0, (len(y) - self.seg_samples) // 2)
            y = y[start:start+self.seg_samples]
        return y.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y_nb_ups = self._load_center_crop(row["nb_ups_path"])   # NB upsampled to 16k (baseline input)
        y_wb = self._load_center_crop(row["wb_path"])            # WB target at 16k

        # Compute mel "images"
        X = wav2mel(y_nb_ups, self.sr, self.n_fft, self.hop, self.win, self.n_mels, self.fmin, self.fmax)
        Y = wav2mel(y_wb, self.sr, self.n_fft, self.hop, self.win, self.n_mels, self.fmin, self.fmax)

        # [C=1, H, W]
        return torch.from_numpy(X[None, ...]), torch.from_numpy(Y[None, ...])
