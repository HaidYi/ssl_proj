import os
import logging
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter

def make_dir(dirname):
    if not os.path.exists(path=dirname):
        os.makedirs(dirname)


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    return datetime.today().strftime(fmt)


def setup_logger(args, log_level=logging.INFO):
    format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    logdir = os.path.join(args.log_dir, f'{args.dset}@{args.n_labeled}_{args.augtype}')
    make_dir(logdir)

    writer = SummaryWriter(logdir=logdir, comment=f'{args.dset}@{args.n_labeled}')

    logging.basicConfig(
        filename=os.path.join(logdir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(format))

    logger = logging.getLogger('train')
    logger.addHandler(console)

    return logger, writer


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_cmd():
    parser = argparse.ArgumentParser("ssvae_fixmatch")

    # model
    parser.add_argument(
        '--model', type=str, choices=['wide_resnet', 'resnet'],
        help='the classification model'
    )
    parser.add_argument('--wresnet_n', type=int, default=28, help='depth of classifying net')
    parser.add_argument('--wresnet_k', type=int, default=2, help='net widen factor')
    parser.add_argument('--drop_rate', type=float, default=0, help='drop out rate in wide resnet')
    parser.add_argument('--z_dim', type=int, default=128, help='dimension of latent var: z')
    parser.add_argument('--threshold', type=float, default=0.95, help='pseudo label threshold')
    parser.add_argument('--mu', type=float, default=2, help='loss balance parameter')

    # dataset
    parser.add_argument(
        '--dset', type=str, default='cifar10',
        choices=['cifar10', 'cifar100', 'svhn'], help='the dataset to use'
    )
    parser.add_argument('--n_class', type=int, default=10, help='num of classes')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--n_labeled', type=int, default=1000, help='labeled samples')
    parser.add_argument('--nc', type=int, default=3, help='channels of input image')

    # training
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of cpus used to feed data')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 in Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 in Adam optimizer')
    # parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='epochs to train model')

    # util
    parser.add_argument('--log_dir', type=str, default='./experiment', help='path to the log dir')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='path to save the model')
    parser.add_argument('--use_cuda', default=True, action='store_true', help='whether to use gpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--augtype', type=str, default='strong', help='include weak or strong augmentation')
    parser.add_argument('--gen', default=True, help='generate images or not')

    args = parser.parse_args()

    return args
