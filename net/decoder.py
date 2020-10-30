import torch
import numpy as np
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, z_dim=128, base_planes=32, out_size=32, out_channels=3):
        super(Decoder, self).__init__()

        num_blocks = int(np.log2(out_size // 4))
        base_planes = base_planes * (2**num_blocks)
        layers = []

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(z_dim, base_planes, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(base_planes),
                nn.ReLU(True)
            )
        )

        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(base_planes, base_planes // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_planes // 2),
                nn.ReLU(True)
            ))
            base_planes = base_planes // 2

        layers.append(nn.Conv2d(base_planes, out_channels, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z)
