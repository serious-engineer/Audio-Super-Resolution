import os
from typing import Callable, Tuple
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F


class AudioSRDataset(Dataset):
    """
    Dataset that creates (low-res mel, high-res mel) pairs from input files.
    Folder structure:
        data/
          hr/file1.wav
          low_sr/file1.wav
    The filenames in hr/ and low_sr/ should match.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int,
        mel_transform: Callable,
        segment_seconds: float = 2.0,
        mono: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.hr_dir = os.path.join(root_dir, "hr")
        self.low_dir = os.path.join(root_dir, "low_sr")
        self.target_sr = target_sr
        self.segment_seconds = segment_seconds
        self.mel_transform = mel_transform
        self.mono = mono

        if not os.path.isdir(self.hr_dir):
            raise RuntimeError(f"High-res dir not found: {self.hr_dir}")
        if not os.path.isdir(self.low_dir):
            raise RuntimeError(f"Low_sr dir not found: {self.low_dir}")

        # Build list of files that exist in both hr and low_sr
        hr_files = [
            f for f in os.listdir(self.hr_dir)
            if f.lower().endswith((".wav", ".flac", ".ogg"))
        ]
        if not hr_files:
            raise RuntimeError(f"No audio files found in {self.hr_dir}")

        self.pairs = []
        for fn in hr_files:
            low_path = os.path.join(self.low_dir, os.path.splitext(fn)[0] + ".wav")
            if os.path.isfile(low_path):
                self.pairs.append((os.path.join(self.hr_dir, fn), low_path))

        if not self.pairs:
            raise RuntimeError(f"No matching hr/low_sr pairs found in {root_dir}")

        self.segment_samples = int(segment_seconds * target_sr)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if self.mono and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav  # [1, T]

    def _random_crop(self, wav_hr: torch.Tensor, wav_low: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, T_hr = wav_hr.shape
        _, T_low = wav_low.shape
        T = min(T_hr, T_low)

        if T <= self.segment_samples:
            pad = self.segment_samples - T
            wav_hr = F.pad(wav_hr[..., :T], (0, pad))
            wav_low = F.pad(wav_low[..., :T], (0, pad))
            return wav_hr, wav_low
        else:
            start = torch.randint(0, T - self.segment_samples, (1,)).item()
            end = start + self.segment_samples
            return wav_hr[:, start:end], wav_low[:, start:end]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_path, low_path = self.pairs[idx]

        wav_hr = self._load_audio(hr_path)
        wav_low = self._load_audio(low_path)
        wav_hr, wav_low = self._random_crop(wav_hr, wav_low)

        # Normalize
        for w in (wav_hr, wav_low):
            max_val = w.abs().max()
            if max_val > 0:
                w /= max_val

        # Compute log-mels
        mel_hr = self.mel_transform(wav_hr) 
        mel_low = self.mel_transform(wav_low)

        mel_hr = mel_hr.squeeze(0) 
        mel_low = mel_low.squeeze(0)

        return mel_low, mel_hr
