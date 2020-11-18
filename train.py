import os
import time
import torch
import shutil
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, RandomSampler
from itertools import cycle
from tqdm import tqdm
from model import ssvae_fixmatch
from util import parse_cmd, setup_logger, AverageMeter, make_dir
from datasets.get_data import get_cifar10, get_cifar100
import matplotlib as mpl
mpl.use('Agg')

DATASET_GETTERS = {
    'cifar10': get_cifar10,
    'cifar100': get_cifar100
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def save_checkpoint(state, is_best, checkpoint, dset, n_labeled, augtype, filename='checkpoint.pth.tar'):
    file_path = os.path.join(checkpoint, filename)
    torch.save(state, file_path)
    if is_best:
        best_model_filename = f"{dset}-{n_labeled}_{augtype}_model_best.pth.tar"
        shutil.copyfile(file_path, os.path.join(checkpoint, best_model_filename))

def test(data_loader, model, args):
    model.eval()
    correct_num = 0
    total_samples = 0

    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            logits = model.encode_y(x)
        pred_label = torch.argmax(logits, dim=-1)
        v = (pred_label == y).sum()

        correct_num += v.item()
        total_samples += x.size(0)
        
    acc = float(correct_num) / total_samples
    return acc

def generate_img(args, img_num = 10,
                 best_model_dir = "./checkpoint",
                 save_dir = "checkpoint"
                ):
    if args.dset == "cifar10":
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif args.dset == "cifar100":
        labels = ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout"]
        # labels = ["beaver", "dolphin", "otter", "seal", "whale",
        #           "aquarium fish", "flatfish", "ray", "shark", "trout",
        #           "orchids", "poppies", "roses", "sunflowers", "tulips",
        #           "bottles", "bowls", "cans", "cups", "plates",
        #           "apples", "mushrooms", "oranges", "pears", "sweet peppers",
        #           "clock", "computer keyboard", "lamp", "telephone", "television",
        #           "bed", "chair", "couch", "table", "wardrobe",
        #           "bee", "beetle", "butterfly", "caterpillar", "cockroach",
        #           "bear", "leopard", "lion", "tiger", "wolf",
        #           "bridge", "castle", "house", "road", "skyscraper",
        #           "cloud", "forest", "mountain", "plain", "sea",
        #           "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
        #           "fox", "porcupine", "possum", "raccoon", "skunk",
        #           "crab", "lobster", "snail", "spider", "worm",
        #           "baby", "boy", "girl", "man", "woman",
        #           "crocodile", "dinosaur", "lizard", "snake", "turtle",
        #           "hamster", "mouse", "rabbit", "shrew", "squirrel",
        #           "maple", "oak", "palm", "pine", "willow",
        #           "bicycle", "bus", "motorcycle", "pickup truck", "train",
        #           "lawn-mower", "rocket", "streetcar", "tank", "tractor"]
    best_model_filename = f"{args.dset}-{args.n_labeled}_{args.augtype}_model_best.pth.tar"
    checkpoint_dir_filename = os.path.join(best_model_dir, best_model_filename)
    model = ssvae_fixmatch(args)
    model = model.to(args.device)
    checkpoint = torch.load(checkpoint_dir_filename)
    model.load_state_dict(checkpoint['state_dict'])
    
    label_num = len(labels)
    labels_list = []
    for labels_i in range(label_num):
        labels_list = labels_list + [labels_i] * img_num
    target = torch.tensor(labels_list)
    one_hot = torch.nn.functional.one_hot(target, args.n_class)
    one_hot = one_hot.to(args.device).float()
    img_gen = model.generate_sample(one_hot)
    
    fig, axs = plt.subplots(label_num, img_num, sharex=True, sharey=True, figsize=(15,15))
    fig.subplots_adjust(wspace=0.075, hspace=0.075)
    for axs_x in range(img_num):
        for axs_y in range(label_num):
            img_idx = img_num * axs_y + axs_x
            img_ij = img_gen.cpu().detach().numpy()
            img_ndarray_ij = img_ij[img_idx, :, :, :].transpose((1, 2, 0))
            axs[axs_y, axs_x].imshow(img_ndarray_ij)
            axs[axs_y, axs_x].axis('off')
            if axs_x == 0:
                label_i = labels[axs_y]
                axs[axs_y, axs_x].text(-1, 0.5, label_i, horizontalalignment='center',
                                       verticalalignment='center', transform=axs[axs_y, axs_x].transAxes, fontsize = 25)
    fig_filename = f"{args.dset}-{args.n_labeled}_{args.augtype}_generation.pdf"
    fig_dir_filename = os.path.join(save_dir, fig_filename)
    plt.savefig(fig_dir_filename, bbox_inches='tight')
    plt.close()

def train(args, labeled_trainloader, unlabeled_trainloader, model: ssvae_fixmatch, optimizer, epoch):
    losses = AverageMeter()
    sup_losses = AverageMeter()
    unsup_losses = AverageMeter()
    fixmatch_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    train_loader = zip(cycle(labeled_trainloader), unlabeled_trainloader)
    model.train()

    n_iter = len(unlabeled_trainloader)
    p_bar = tqdm(range(n_iter))

    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        inputs_x, targets_x = data_x
        (inputs_u, inputs_u_w, inputs_u_s), _ = data_u
        # move data to gpu
        inputs_x = inputs_x.to(args.device)
        targets_x = targets_x.to(args.device)
        inputs_u = inputs_u.to(args.device)
        inputs_u_w = inputs_u_w.to(args.device)
        inputs_u_s = inputs_u_s.to(args.device)

        data_time.update(time.time() - end)

        sup_loss, unsup_loss, fixmatch_loss = model.compute_loss(
            inputs_x, targets_x, inputs_u, inputs_u_w, inputs_u_s
        )

        # compute total loss
        loss = sup_loss + unsup_loss + args.mu * fixmatch_loss

        loss.backward()
        optimizer.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        sup_losses.update(sup_loss.item())
        unsup_losses.update(unsup_loss.item())
        fixmatch_losses.update(fixmatch_loss.item())

        p_bar.set_description('Training Epoch: {epoch}/{epochs:4}, Iter: {batch:4}/{iter:4}, Data: {data:.3f}s, Batch: {bt:.3f}s, loss: {loss:.3f}, sup_loss: {sup_loss:.3f}, unsup_loss: {unsup_loss:.3f}, fixmatch_loss: {fixmatch_loss:.3f}'.format(
            epoch=epoch + 1,
            epochs=args.n_epochs,
            batch=batch_idx + 1,
            iter=n_iter,
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            sup_loss=sup_losses.avg,
            unsup_loss=unsup_losses.avg,
            fixmatch_loss=fixmatch_losses.avg
        ))
        p_bar.update()

    p_bar.close()

    return losses.avg, sup_losses.avg, unsup_losses.avg, fixmatch_losses.avg

def main():
    # parse command
    args = parse_cmd()

    # set up logging
    logger, writer = setup_logger(args)
    make_dir(args.checkpoint)
    
    # set up random seed
    if args.seed is not None:
        set_seed(args)
    
    # set up model config
    if args.dset == 'cifar10' or args.dset == 'cifar100':
        args.img_size = 32
        args.nc = 3
        args.n_class = int(args.dset[5:])
        if args.model == 'wide_resnet':
            args.wresnet_n = 28
            args.wresnet_k = 2
        if args.model == 'resnet':
            pass
    
    logger.info(dict(args._get_kwargs()))
    
    # set up datasets and dataloader
    labeled_dataset, unlabled_dataset, test_dataset = DATASET_GETTERS[f'{args.dset}'](
        './data', args.n_labeled)
    
    labeled_train_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(labeled_dataset),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    unlabeled_trainloader = DataLoader(
        unlabled_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(unlabled_dataset),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # create the model and move to gpu
    model = ssvae_fixmatch(args)
    model = model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    
    logger.info("***** Start Training *****")
    logger.info(f"  Task = {args.dset}@{args.n_labeled}_{args.augtype}")
    logger.info(f"  Num Epochs = {args.n_epochs}")
    logger.info(f"  Batch Size = {args.batch_size}")
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    
    test_accs = []
    best_acc = 0
    model.zero_grad()
    
    best_log_filename = f"log_best_{args.dset}-{args.n_labeled}_{args.augtype}.txt"
    best_log_dir_filename = os.path.join(args.log_dir, f"{args.dset}@{args.n_labeled}_{args.augtype}", best_log_filename)
    with open(best_log_dir_filename, 'w+') as best_log:
        for epoch in range(args.n_epochs):
        
            loss, sup_loss, unsup_loss, fixmatch_loss = train(
                args, labeled_train_loader, unlabeled_trainloader, model, optimizer, epoch)
        
            test_acc = test(test_loader, model, args)
        
            logger.info("Epoch {:3d}, loss: {:.3f}, sup_loss: {:.3f}, unsup_loss: {:.3f}, fixmatch_loss: {:.3f}, test_acc: {:.3f}".format(
                epoch + 1, loss, sup_loss, unsup_loss, fixmatch_loss, test_acc
            )
                       )
        
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/sup_loss', sup_loss, epoch)
            writer.add_scalar('train/unsup_loss', unsup_loss, epoch)
            writer.add_scalar('train/fixmatch_loss', fixmatch_loss, epoch)
            writer.add_scalar('test/test_acc', test_acc, epoch)
        
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            test_accs.append(test_acc)
        
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer
            }, is_best, args.checkpoint, args.dset, args.n_labeled, args.augtype,
                filename=f'checkpoint_{args.dset}@{args.n_labeled}_{args.augtype}.pth.tar')
            
            if is_best:
                best_log.write("Epoch {:3d}, loss: {:.3f}, sup_loss: {:.3f}, unsup_loss: {:.3f}, fixmatch_loss: {:.3f}, test_acc: {:.3f}\n".format(
                epoch + 1, loss, sup_loss, unsup_loss, fixmatch_loss, test_acc))
                
        writer.close()
    
    if args.gen == True:
        generate_img(args, img_num = 10)

if __name__ == "__main__":
    main()
