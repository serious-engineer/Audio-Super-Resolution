# src/train.py
# GAN with Generator and Discriminator training script for Audio Super-Resolution on mel-spectrograms.

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from pathlib import Path
from src.models.audio_srgan import AudioSRGenerator, AudioSRDiscriminator
from src.utils.mel_utils import LogMelExtractor
from src.audio_dataset import AudioSRDataset
from src.losses.bae_losses import MelLoss

# training function
def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #  Mel extractor 
    log_mel = LogMelExtractor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        apply_log=False, 
    )
    print("[Mel] LogMelExtractor initialized.")
    #  Dataloader
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
    print(f"[Data] Loaded dataset with {len(dataset)} samples.")
    #  Generator & Discriminator
    G = AudioSRGenerator(num_res_blocks=args.num_res_blocks).to(device)
    D = AudioSRDiscriminator().to(device)

    print(f"[Model] Generator params: {sum(p.numel() for p in G.parameters())/1e6:.2f} M")
    print(f"[Model] Discriminator params: {sum(p.numel() for p in D.parameters())/1e6:.2f} M")

    if args.g_ckpt:
        g_ckpt_path = Path(args.g_ckpt)
        if g_ckpt_path.is_file():
            print(f"[Init] Loading generator weights from {g_ckpt_path}")
            state = torch.load(g_ckpt_path, map_location=device)
            if isinstance(state, dict) and "G" in state:
                G.load_state_dict(state["G"])
            else:
                G.load_state_dict(state)
        else:
            print(f"[Init] WARNING: g_ckpt not found at {g_ckpt_path}, starting from scratch.")

    #  Optimizer
    optimizer_G = optim.Adam(
        G.parameters(),
        lr=args.lr_g,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    optimizer_D = optim.Adam(
        D.parameters(),
        lr=args.lr_d,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    # Reconstruction loss in mel domain (from BAE Reference Paper )
    mel_loss = MelLoss(
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        lambda_sc=args.lambda_sc,
    )
    print("[Loss] MelLoss initialized.")
    #  Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if args.resume_ckpt:
            ckpt_path = Path(args.resume_ckpt)
        else:
            ckpt_path = out_dir / "ckpt_last.pt"

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"[Resume] Checkpoint not found: {ckpt_path}")

        print(f"[Resume] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Load generator
        if "G" in checkpoint:
            G.load_state_dict(checkpoint["G"])
        else:
            print("[Resume] WARNING: 'G' not found in checkpoint.")

        # Load discriminator
        if "D" in checkpoint:
            try:
                D.load_state_dict(checkpoint["D"])
            except Exception as e:
                print(f"[Resume] WARNING: Could not load D from checkpoint: {e}")
        else:
            print("[Resume] WARNING: 'D' not found in checkpoint. D will be re-initialized.")

        # Load optimizers 
        if "optG" in checkpoint:
            optimizer_G.load_state_dict(checkpoint["optG"])
        else:
            print("[Resume] WARNING: 'optG' not in checkpoint.")

        if "optD" in checkpoint:
            optimizer_D.load_state_dict(checkpoint["optD"])
        else:
            print("[Resume] WARNING: 'optD' not in checkpoint.")

        last_epoch = checkpoint.get("epoch", -1)
        start_epoch = int(last_epoch) + 1
        print(f"[Resume] Resuming from epoch {start_epoch} (last finished epoch = {last_epoch})")

    #  Training loop
    step = 0
    G.train()
    D.train()

    for epoch in range(start_epoch, args.epochs):
        print(f"========== Epoch {epoch} / {args.epochs - 1} ==========")
        for batch_idx, (mel_low, mel_hr) in enumerate(dataloader):
            mel_low = mel_low.to(device, non_blocking=True)  
            mel_hr = mel_hr.to(device, non_blocking=True)    

            # Update Discriminator
            optimizer_D.zero_grad(set_to_none=True)

            with torch.no_grad():
                mel_sr_detached = G(mel_low)

            # Real / fake scores
            d_real = D(mel_hr)
            d_fake = D(mel_sr_detached)

            # loss for D
            loss_d_real = torch.relu(1.0 - d_real).mean()
            loss_d_fake = torch.relu(1.0 + d_fake).mean()
            d_loss = loss_d_real + loss_d_fake

            d_loss.backward()
            if args.grad_clip > 0.0:
                clip_grad_norm_(D.parameters(), args.grad_clip)
            optimizer_D.step()

            # Update Generator
            optimizer_G.zero_grad(set_to_none=True)

            mel_sr = G(mel_low)

            # Reconstruction loss
            rec_loss, rec_stats = mel_loss(mel_sr, mel_hr)

            # Adversarial loss for G
            d_fake_for_g = D(mel_sr)
            g_adv_loss = -d_fake_for_g.mean()

            # feature-matching loss on the last feature map
            fm_loss = torch.tensor(0.0, device=device)
            if args.lambda_fm > 0.0:
                with torch.no_grad():
                    feat_real = D.features(mel_hr)
                feat_fake = D.features(mel_sr)
                fm_loss = torch.mean(torch.abs(feat_fake - feat_real))

            # Total generator loss
            g_loss = (
                args.lambda_rec * rec_loss +
                args.lambda_adv * g_adv_loss +
                args.lambda_fm * fm_loss
            )

            g_loss.backward()
            if args.grad_clip > 0.0:
                clip_grad_norm_(G.parameters(), args.grad_clip)
            optimizer_G.step()

            # Logging
            if step % args.log_interval == 0:
                print(
                    f"[Ep {epoch:03d} | Step {step:06d}] "
                    f"D={d_loss.item():.4f} "
                    f"G={g_loss.item():.4f} "
                    f"Rec={rec_loss.item():.4f} "
                    f"Adv={g_adv_loss.item():.4f} "
                    f"FM={fm_loss.item():.4f} "
                    f"L1={rec_stats.get('mel_l1', 0.0):.4f} "
                    f"MSE={rec_stats.get('mel_mse', 0.0):.4f} "
                    f"SC={rec_stats.get('mel_sc', 0.0):.4f}"
                )

            step += 1

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optG": optimizer_G.state_dict(),
            "optD": optimizer_D.state_dict(),
        }
        torch.save(ckpt, out_dir / "ckpt_last.pt")
        print(f"[Checkpoint] Saved ckpt_last.pt at end of epoch {epoch}")

    # Final generator checkpoint for inference
    torch.save(G.state_dict(), out_dir / "generator_final.pt")
    print("[Checkpoint] Saved generator_final.pt")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AudioSR model with GAN (Generator + Discriminator) on mel-spectrograms."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root directory with hr/ and low_sr/ subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where checkpoints will be saved.",
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Target sample rate for training (e.g., 22050 or 48000).",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Number of mel bands.",
    )
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=2.0,
        help="Random segment length in seconds for training.",
    )

    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=16,
        help="Number of residual blocks in the generator.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=2e-4,
        help="Learning rate for generator.",
    )
    parser.add_argument(
        "--lr_d",
        type=float,
        default=2e-4,
        help="Learning rate for discriminator.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max-norm (0 to disable).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Steps between console log prints.",
    )

    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=1.0,
        help="Weight for L1 term inside MelLoss.",
    )
    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=1.0,
        help="Weight for L2/MSE term inside MelLoss.",
    )
    parser.add_argument(
        "--lambda_sc",
        type=float,
        default=0.5,
        help="Weight for spectral convergence term inside MelLoss.",
    )

    parser.add_argument(
        "--lambda_rec",
        type=float,
        default=1.0,
        help="Global weight for mel reconstruction loss.",
    )
    parser.add_argument(
        "--lambda_adv",
        type=float,
        default=0.1,
        help="Weight for adversarial loss term.",
    )
    parser.add_argument(
        "--lambda_fm",
        type=float,
        default=1.0,
        help="Weight for feature-matching loss term.",
    )

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

    parser.add_argument(
        "--g_ckpt",
        type=str,
        default="",
        help="Path to generator checkpoint to initialize from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
