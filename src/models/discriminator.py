import torch
import torch.nn as nn

def d_block(in_ch, out_ch, k=4, s=2, p=1, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p), nn.LeakyReLU(0.2, inplace=True)]
    if norm:
        layers.insert(1, nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            d_block(in_ch, base, norm=False),
            d_block(base, base*2),
            d_block(base*2, base*4),
            d_block(base*4, base*8),
            nn.Conv2d(base*8, 1, 3, 1, 1)  # patch logits
        )
    def forward(self, x):
        return self.net(x)
