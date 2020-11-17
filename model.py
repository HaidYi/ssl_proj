import torch
import torch.nn as nn
import torch.nn.functional as F

from net.wide_resnet import Wide_ResNet
from net.wideresnet import WideResNet
from net.encoder import encoder_x, MLP
from net.decoder import Decoder
from net.util import Flatten
import argparse


def reparametrize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(mu)
    return mu + eps * std


class ssvae_fixmatch(nn.Module):
    def __init__(self, args):
        super(ssvae_fixmatch, self).__init__()

        self.n_class = args.n_class
        self.z_dim = args.z_dim
        self.data_size = args.img_size ** 2 * args.nc
        self.device = args.device
        self.use_cuda = args.use_cuda
        self.args = args

        # initialize net
        self.encoder_y = WideResNet(
            depth=args.wresnet_n, widen_factor=args.wresnet_k,
            drop_rate=args.drop_rate, num_classes=args.n_class)
        self.decoder = Decoder(args.z_dim + args.n_class, out_size=args.img_size)

        # create self.encoder_z and self.encoder_x
        self.encoder_z = MLP(
            i_dim=self.z_dim + self.n_class,
            h_dims=(self.z_dim * 3, ) + (self.z_dim * 2, ),
            non_linear=nn.Softplus,
            out_activation=None
        )
        self.encoder_x = encoder_x(
            args.nc, [32, 32, 64, 128], z_dim=args.z_dim, input_size=args.img_size
        )

    def encode_z(self, x, y):
        '''
        q(z|x, y) = Normal(loc(x,y), scale(x,y)) # encode the latent var z
        '''
        x_y = torch.cat([x, y], dim=-1)
        h = self.encoder_z(x_y)
        mu, logvar = torch.split(h, self.z_dim, dim=-1)
        return mu, logvar

    def encode_y(self, x):
        '''
        q(y|x) = Categorical(logits(x)) # predict the label y from x
        '''
        logit = self.encoder_y(x)
        return logit

    def decode(self, y, z):
        '''
        p(x|y, z) = Decoder(y, z)
        '''
        y_z = torch.cat([y, z], dim=-1)
        recons_x = self.decoder(y_z.view(-1, self.z_dim + self.n_class, 1, 1))
        return recons_x

    def kl_div(self, mu, logvar):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return klds

    def mse_loss(self, recons_x, x):
        recon_loss = F.mse_loss(recons_x, x, reduction='none')
        return recon_loss

    def sup_loss(self, x, y):
        hx = self.encoder_x(x)
        y = F.one_hot(y, num_classes=self.n_class).float()

        mu, logvar = self.encode_z(hx, y)
        z = reparametrize(mu, logvar)
        recons_x = self.decode(y, z)

        recon_loss = self.mse_loss(recons_x, x).view(x.size(0), -1).sum(dim=-1)
        kl_div = self.kl_div(mu, logvar).sum(dim=-1)

        sup_loss = recon_loss + kl_div
        sup_loss = sup_loss.mean() / self.data_size

        return sup_loss

    def unsup_loss(self, u):
        batch_size = u.size(0)
        pi = self.encoder_y(u).softmax(dim=-1)

        hu = self.encoder_x(u)
        hu = hu.unsqueeze(dim=1).repeat(1, self.n_class, 1)
        y = torch.arange(self.n_class)
        target_u = u.reshape(batch_size, -1).unsqueeze(dim=1).repeat(1, self.n_class, 1)

        if self.use_cuda:
            y = y.to(self.device)
        y = F.one_hot(y, num_classes=self.n_class).float()
        y = y.unsqueeze(dim=0).repeat(hu.size(0), 1, 1)

        mu, logvar = self.encode_z(hu, y)
        z = reparametrize(mu, logvar)
        recons_u = self.decode(y.reshape(-1, self.n_class), z.reshape(-1, self.z_dim))
        recons_u = recons_u.view(batch_size * self.n_class, -1).view(batch_size, self.n_class, -1)

        # TO DO: change the recon_loss
        recon_loss = self.mse_loss(recons_u, target_u).sum(dim=-1)
        kl_div = self.kl_div(mu, logvar).sum(dim=-1)

        entropy = (- pi * torch.log(pi + 1e-8)).sum(dim=-1)
        unsup_loss = ((recon_loss + kl_div) * pi).sum(dim=-1) - entropy

        unsup_loss = unsup_loss.mean() / self.data_size

        return unsup_loss

    def fixmatch_loss(self, x, y, u, u_w, u_s):
        inputs = torch.cat([x, u_w, u_s], dim=0)
        logits = self.encode_y(inputs)
        logits_x = logits[:self.args.batch_size]
        logits_u_w, logits_u_s = logits[self.args.batch_size:].chunk(2)

        Lx = F.cross_entropy(logits_x, y, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs, targets_u_s = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.args.threshold).float()
        
        Lu = (F.cross_entropy(logits_u_s, targets_u_s, reduction='none') * mask).mean()
        if self.args.augtype == "strong":
#             print("Apply strong augmentation.")
            return Lx + Lu
        elif self.args.augtype == "weak":
#             print("Apply weak augmentation.")
            return Lx
        
    def compute_loss(self, x, y, u, u_w, u_s):
        sup_loss = self.sup_loss(x, y)
        unsup_loss = self.unsup_loss(u)
        classify_loss = self.fixmatch_loss(x, y, u, u_w, u_s)

        return sup_loss, unsup_loss, classify_loss

    def generate_sample(self, y):
        z = torch.randn(y.size(0), self.z_dim)
        z = z.to(self.args.device)
        return self.decode(y, z)

def dict2namespace(dict_):
    namespace = argparse.Namespace()
    for key, value in dict_.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


if __name__ == "__main__":
    args = {
        'batch_size': 32,
        'z_dim': 128,
        'use_cuda': False,
        'device': 'cpu',
        'wresnet_n': 28,
        'wresnet_k': 2,
        'drop_rate': 0,
        'n_class': 10,
        'img_size': 32,
        'nc': 3,
        'threshold': 0.95
    }

    args = dict2namespace(args)

    model = ssvae_fixmatch(args)

    x = torch.randn(32, 3, 32, 32)
    u = torch.randn(32, 3, 32, 32)
    u_w = torch.randn(32, 3, 32, 32)
    u_s = torch.randn(32, 3, 32, 32)
    y = torch.empty(32, dtype=torch.long).random_(10)

    sup_loss, unsup_loss, fixmatch_loss = model.compute_loss(x, y, u, u_w, u_s)

    msg = f"sup_loss: {sup_loss.item():.3f}, unsup_loss: {unsup_loss.item():.3f}, fixmatch_loss: {fixmatch_loss.item():.3f}"
    print(msg)
