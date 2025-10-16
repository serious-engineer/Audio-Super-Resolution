import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )

class UNetGen(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        # encoder
        self.e1 = conv_block(in_ch, base)
        self.e2 = conv_block(base, base*2, s=2)
        self.e3 = conv_block(base*2, base*4, s=2)
        self.e4 = conv_block(base*4, base*8, s=2)
        # bottleneck
        self.b = conv_block(base*8, base*8)
        # decoder
        self.d4 = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1)
        self.c4 = conv_block(base*8, base*4)
        self.d3 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.c3 = conv_block(base*4, base*2)
        self.d2 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.c2 = conv_block(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.b(e4)
        u4 = self.d4(b)
        u4 = self.c4(torch.cat([u4, e3], dim=1))
        u3 = self.d3(u4)
        u3 = self.c3(torch.cat([u3, e2], dim=1))
        u2 = self.d2(u3)
        u2 = self.c2(torch.cat([u2, e1], dim=1))
        return self.out(u2)  # logits in log-mel space
