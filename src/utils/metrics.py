import torch
import torchaudio
import torch.nn.functional as F

try:
    from pystoi.stoi import stoi
    from pesq import pesq
    HAS_PYSTOI = True
    HAS_PESQ = True
except Exception:
    HAS_PYSTOI = False
    HAS_PESQ = False


def align_signals(ref: torch.Tensor, est: torch.Tensor):
    ref = ref.squeeze()
    est = est.squeeze()
    min_len = min(ref.shape[-1], est.shape[-1])
    return ref[..., :min_len], est[..., :min_len]


def snr_db(ref: torch.Tensor, est: torch.Tensor, use_alignment: bool = True) -> float:
    if ref.dim() > 1:
        ref = ref.mean(dim=0)
    if est.dim() > 1:
        est = est.mean(dim=0)

    if use_alignment:
        ref_aligned, est_aligned = align_signals(ref, est)
    else:
        ref_aligned, est_aligned = ref, est

    ref_aligned = ref_aligned.float().cpu()
    est_aligned = est_aligned.float().cpu()

    L = min(ref_aligned.numel(), est_aligned.numel())
    ref_aligned = ref_aligned[:L]
    est_aligned = est_aligned[:L]

    scale = torch.dot(ref_aligned, est_aligned) / (est_aligned.pow(2).sum() + 1e-12)
    est_aligned = est_aligned * scale

    noise = ref_aligned - est_aligned
    signal_power = ref_aligned.pow(2).mean()
    noise_power = noise.pow(2).mean() + 1e-12

    snr = 10 * torch.log10(signal_power / noise_power)
    return float(snr.item())

def seg_snr_db(ref, est, sr=22050, frame_ms=20, hop_ms=10,min_db=-10.0, max_db=35.0):
    if ref.dim() > 1:
        ref = ref.mean(dim=0)
    if est.dim() > 1:
        est = est.mean(dim=0)

    L = min(ref.numel(), est.numel())
    ref = ref[:L]
    est = est[:L]

    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    pad = frame_len - (L - frame_len) % hop_len
    ref = F.pad(ref, (0, pad))
    est = F.pad(est, (0, pad))

    ref_frames = ref.unfold(0, frame_len, hop_len)
    est_frames = est.unfold(0, frame_len, hop_len)

    ref_pow = (ref_frames ** 2).sum(dim=1)
    noise_pow = ((ref_frames - est_frames) ** 2).sum(dim=1) + 1e-12
    snr_frames = 10 * torch.log10(ref_pow / noise_pow)
    snr_frames = torch.clamp(snr_frames, min_db, max_db)
    return float(snr_frames.mean().item())

def mse(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref, est = align_signals(ref, est)
    return float(F.mse_loss(est, ref).item())


def mae(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref, est = align_signals(ref, est)
    return float(F.l1_loss(est, ref).item())


def log_spectral_distance(
    ref: torch.Tensor,
    est: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 480,
    eps: float = 1e-8,
) -> float:
    ref, est = align_signals(ref, est)
    win = torch.hann_window(n_fft, device=ref.device)
    ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=hop_length,
                          window=win, return_complex=True)
    est_spec = torch.stft(est, n_fft=n_fft, hop_length=hop_length,
                          window=win, return_complex=True)

    ref_mag = ref_spec.abs().clamp_min(eps)
    est_mag = est_spec.abs().clamp_min(eps)

    log_ref = torch.log(ref_mag)
    log_est = torch.log(est_mag)

    lsd_per_frame = (log_ref - log_est).pow(2).mean(dim=0).sqrt()
    return float(lsd_per_frame.mean().item())


def compute_stoi_pesq(
    ref: torch.Tensor,
    est: torch.Tensor,
    sr_ref: int,
    target_sr: int = 16000,
):
    if not (HAS_PYSTOI and HAS_PESQ):
        print("[metrics] pystoi/pesq not available, returning (None, None)")
        return None, None

    if ref.dim() > 1:
        ref = ref.mean(dim=0)
    if est.dim() > 1:
        est = est.mean(dim=0)

    ref = ref.float().cpu()
    est = est.float().cpu()

    print(f"[metrics] compute_stoi_pesq: sr_ref={sr_ref}, target_sr={target_sr}")
    print(f"[metrics] ref shape={ref.shape}, est shape={est.shape}")

    try:
        ref_rs = torchaudio.functional.resample(ref.unsqueeze(0), sr_ref, target_sr).squeeze(0)
        est_rs = torchaudio.functional.resample(est.unsqueeze(0), sr_ref, target_sr).squeeze(0)
    except Exception as e:
        print(f"[metrics] Resample error: {e}")
        return None, None

    min_len = min(ref_rs.numel(), est_rs.numel())
    ref_rs = ref_rs[:min_len]
    est_rs = est_rs[:min_len]

    ref_np = ref_rs.numpy()
    est_np = est_rs.numpy()

    stoi_val, pesq_val = None, None
    try:
        stoi_val = float(stoi(ref_np, est_np, target_sr, extended=False))
    except Exception as e:
        print(f"[metrics] STOI error: {e}")
        stoi_val = None

    try:
        pesq_val = float(pesq(target_sr, ref_np, est_np, "wb"))
    except Exception as e:
        print(f"[metrics] PESQ error: {e}")
        pesq_val = None

    return stoi_val, pesq_val
