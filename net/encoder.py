import torch
import torch.nn as nn
from net.util import Flatten

class encoder_x(nn.Module):

    def __init__(self, in_channels, hidden_dims, z_dim, input_size):
        super(encoder_x, self).__init__()

        modules = []
        # build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        output_size = input_size // (2 ** len(hidden_dims))
        modules.append(
            nn.Sequential(
                Flatten(),
                nn.Linear(hidden_dims[-1] * (output_size ** 2), z_dim)
            )
        )

        self.main = nn.Sequential(*modules)

    def forward(self, x):
        out = self.main(x)
        return out


class MLP(nn.Module):
    def __init__(self, i_dim, h_dims, non_linear, out_activation):
        super(MLP, self).__init__()
        assert len(h_dims) >= 1, 'h_dims cannot be empty'

        layers = []
        for h_dim in h_dims[:-1]:
            layers.append(nn.Linear(i_dim, h_dim))
            layers.append(non_linear())
            i_dim = h_dim
        layers.append(nn.Linear(i_dim, h_dims[-1]))

        if out_activation:
            layers.append(out_activation())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


if __name__ == "__main__":
    model = encoder_x(in_channels=3, hidden_dims=[32, 32, 64, 128], z_dim=128, input_size=32)
    x = torch.randn(64, 3, 32, 32)
    out = model(x)

    print(out.shape)
