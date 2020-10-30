import math
import torch
import torch.nn as nn
import torch.nn.init as init


init_mtd = {
    'xavier': lambda w, g: init.xavier_normal_(w, gain=g),
    'kaiming': lambda w, m, a: init.kaiming_uniform_(w, mode=m, nonlinearity=a),
    'constant': lambda w, c: init.constant_(w, val=c)
}


def conv3x3(dim_in, dim_out, stride=1, bias=True):
    return nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x1(dim_in, dim_out, stride=1, bias=True):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, padding=0, bias=bias)


def weight_init(module, type='xavier', **args):
    assert isinstance(module, nn.Module), "input module is not an nn.Module"
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        if type == 'xavier':
            init_mtd['xavier'](module.weight, math.sqrt(2))
        elif type == 'kaiming':
            init_mtd['kaiming'](module.weight, args['mode'], args['nonlinearity'])
        else:
            raise NotImplementedError('The input init method is unknown')
    elif classname.find('BatchNorm') != -1:
        init_mtd['constant'](module.weight, 1)
        init_mtd['constant'](module.weight, 0)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
