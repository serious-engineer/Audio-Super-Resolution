# src/train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from pathlib import Path
from src.models.audio_srgan import AudioSRGenerator
from src.utils.mel_utils import LogMelExtractor
from src.audio_dataset import AudioSRDataset
from src.losses.bae_losses import MelLoss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    #  Mel extractor
    log_mel = LogMelExtractor(
        sample_rate=args.sample_rate,  # must be 22050
        n_mels=args.n_mels,           # must be 80
        # n_fft/hop/padding are forced internally to 1024/256/"same"
        apply_log=False,
    )

    #  DataLoader
    dataset = AudioSRDataset(
        root_dir=args.dataset_dir,
        target_sr=args.sample_rate,
        mel_transform=log_mel,
        segment_seconds=args.segment_seconds,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[Data] Loaded dataset with {len(dataset)} items.")
    #  Model (Generator only)
    G = AudioSRGenerator(num_res_blocks=args.num_res_blocks).to(device)

    if args.g_ckpt and Path(args.g_ckpt).is_file():
        print(f"[Init] Loading generator checkpoint from {args.g_ckpt}")
        state_dict = torch.load(args.g_ckpt, map_location=device)
        if isinstance(state_dict, dict) and "G" in state_dict:
            G.load_state_dict(state_dict["G"])
        else:
            G.load_state_dict(state_dict)

    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #  Resume from ckpt_last.pt
    if args.resume:
        if args.resume_ckpt:
            ckpt_path = args.resume_ckpt
        else:
            ckpt_path = os.path.join(args.out_dir, "ckpt_last.pt")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        print(f"[Resume] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        G.load_state_dict(checkpoint["G"])
        optimizer_G.load_state_dict(checkpoint["optG"])

        last_epoch = checkpoint.get("epoch", 0)
        start_epoch = last_epoch + 1
        print(f"[Resume] Resuming from epoch {start_epoch} (last finished epoch was {last_epoch})")

    #  Loss functions
    mel_loss = MelLoss(
        lambda_l1=1.0,
        lambda_l2=1.0,
        lambda_sc=0.5
    ).to(device)

    step = 0

    #  Training loop (Generator only)
    for epoch in range(start_epoch, args.epochs):
        G.train()

        for mel_lr, mel_hr in dataloader:
            mel_lr = mel_lr.to(device)  
            mel_hr = mel_hr.to(device)  

            if mel_lr.dim() == 5:
                mel_lr = mel_lr.squeeze(1)
                mel_hr = mel_hr.squeeze(1)

            optimizer_G.zero_grad(set_to_none=True)

            # Forward
            mel_fake = G(mel_lr)

            # Reconstruction loss
            g_loss, rec_stats = mel_loss(mel_fake, mel_hr)


            if not torch.isfinite(g_loss):
                print(f"[WARN] Non-finite G loss at step {step}, skipping G update.")
            else:
                g_loss.backward()
                clip_grad_norm_(G.parameters(), max_norm=5.0)
                optimizer_G.step()

            # Logging
            if step % args.log_every == 0:
                print(
                    f"[E{epoch} S{step}] "
                    f"G={g_loss.item():.4f}, "
                    f"L1={rec_stats['mel_l1']:.4f}, "
                    f"MSE={rec_stats['mel_mse']:.4f}, "
                    f"SC={rec_stats['mel_sc']:.4f}"
                )
            step += 1

        # Save ckpt_last
        state = {
            "epoch": epoch,
            "G": G.state_dict(),
            "optG": optimizer_G.state_dict(),
        }
        torch.save(state, os.path.join(args.out_dir, "ckpt_last.pt"))
        print(f"[Checkpoint] Saved ckpt_last.pt at end of epoch {epoch}")

    # generator checkpoint for inference
    torch.save(G.state_dict(), out_dir / "generator_final.pt")
    print("[Checkpoint] Saved generator_final.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AudioSR model on mel-spectrograms (no GAN).")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint (ckpt_last.pt by default).",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Path to checkpoint to resume from (overrides default).",
    )

    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument(
        "--low_sr",
        type=int,
        default=8000,
        help="Low-res simulation sample rate (if you generate low_sr yourself).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--lambda_mel",
        type=float,
        default=1.0,
        help="Weight for L1 mel loss in BAELikeGeneratorLoss.",
    )
    parser.add_argument(
        "--lambda_stft",
        type=float,
        default=0.5,
        help="Weight for multi-res STFT loss in BAELikeGeneratorLoss.",
    )

    parser.add_argument("--num_res_blocks", type=int, default=16)

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument(
        "--g_ckpt",
        type=str,
        default="",
        help="Path to generator checkpoint to initialize from",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)