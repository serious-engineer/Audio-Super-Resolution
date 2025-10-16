import os, argparse, yaml, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.datasets.nbwb_mel import NBWBMelPairs
from src.models.generator import UNetGen
from src.models.discriminator import PatchDiscriminator
from tqdm import tqdm

class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def real(self, x):
        return self.bce(x, torch.ones_like(x))
    def fake(self, x):
        return self.bce(x, torch.zeros_like(x))

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NBWBMelPairs(cfg["pairs_csv"], cfg["sr"], cfg["fft"], cfg["hop"], cfg["win"],
                      cfg["n_mels"], cfg["fmin"], cfg["fmax"], cfg["segment_seconds"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, drop_last=True)

    G = UNetGen().to(device)
    D = PatchDiscriminator().to(device)

    optG = optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    gan_loss = GANLoss().to(device)
    l1 = nn.L1Loss()

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["sample_dir"], exist_ok=True)

    for epoch in range(cfg["epochs"]):
        G.train(); D.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)

            # --- Train D ---
            with torch.no_grad():
                Y_hat = G(X)
            D_real = D(Y)
            D_fake = D(Y_hat.detach())
            loss_D = gan_loss.real(D_real) + gan_loss.fake(D_fake)
            optD.zero_grad(); loss_D.backward(); optD.step()

            # --- Train G ---
            D_fake = D(G(X))
            loss_adv = gan_loss.real(D_fake)
            loss_l1 = l1(G(X), Y)
            loss_G = cfg["lambda_adv"]*loss_adv + cfg["lambda_l1"]*loss_l1
            optG.zero_grad(); loss_G.backward(); optG.step()

            pbar.set_postfix(loss_D=float(loss_D.item()), loss_G=float(loss_G.item()), L1=float(loss_l1.item()))
        # save
        torch.save(G.state_dict(), os.path.join(cfg["checkpoint_dir"], f"G_e{epoch+1}.pt"))
        torch.save(D.state_dict(), os.path.join(cfg["checkpoint_dir"], f"D_e{epoch+1}.pt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()
