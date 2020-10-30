import torch
import torch.nn as nn
import torch.nn.functional as F
from net.util import conv1x1, conv3x3, weight_init


class wrn_block(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate, stride=1):
        super(wrn_block, self).__init__()

        self.bn1 = nn.BatchNorm2d(dim_in)
        self.conv1 = conv3x3(dim_in, dim_out)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        self.bn2 = nn.BatchNorm2d(dim_out)
        self.conv2 = conv3x3(dim_out, dim_out, stride)

        self.shortcut = nn.Sequential()
        if dim_in != dim_out:
            self.shortcut = nn.Sequential(
                conv1x1(dim_in, dim_out, stride))

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.relu(self.bn2(out))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, dim_out, init_type='xavier'):
        super(Wide_ResNet, self).__init__()

        self.dim_in = 16
        assert ((depth - 4) % 6 == 0), 'Wide ResNet depth should be 6n+4'
        n_blocks = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self.wide_layer(wrn_block, nStages[1], n_blocks, dropout_rate, stride=1)
        self.layer2 = self.wide_layer(wrn_block, nStages[2], n_blocks, dropout_rate, stride=2)
        self.layer3 = self.wide_layer(wrn_block, nStages[3], n_blocks, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], dim_out)

        self.init_type = init_type
        self.param_init()

    def param_init(self):
        for m in self.modules():
            weight_init(m, type=self.init_type)

    def wide_layer(self, block, dim, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.dim_in, dim, dropout_rate, stride))
            self.dim_in = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    wrn28_10 = Wide_ResNet(28, 10, 0.2, 10)
    print(wrn28_10)

    x = torch.randn(1, 3, 32, 32)
    out = wrn28_10(x)

    print(out.shape)
