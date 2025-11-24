
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block generators."""
    def __init__(self, num_features: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AudioSRGenerator(nn.Module):
    """Generator operating on mel-spectrograms.
    Input:  [B, 1, n_mels, T]  (log-mel of band-limited audio)
    Output: [B, 1, n_mels, T]  (log-mel with hallucinated high-frequency content)
    """
    def __init__(self, num_res_blocks: int = 16, num_features: int = 64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_res_blocks)]
        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
        )
        self.output = nn.Conv2d(
            num_features,
            1,
            kernel_size=9,
            stride=1,
            padding=4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial = self.initial(x)
        out = self.res_blocks(initial)
        out = self.res_conv(out)
        out = out + initial
        out = self.output(out)
        return out


class AudioSRDiscriminator(nn.Module):
    """Discriminator on mel-spectrograms.
    """
    def __init__(self):
        super().__init__()

        def block(in_channels, out_channels, stride):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return nn.Sequential(*layers)

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        layers.append(block(64, 64, stride=2))
        layers.append(block(64, 128, stride=1))
        layers.append(block(128, 128, stride=2))
        layers.append(block(128, 256, stride=1))
        layers.append(block(256, 256, stride=2))
        layers.append(block(256, 512, stride=1))
        layers.append(block(512, 512, stride=2))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.classifier(feat)
        return out
