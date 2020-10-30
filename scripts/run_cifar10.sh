#!/bin/bash

GPU_ID=3

CUDA_VISIBLE_DEVICES=3 python train.py \
  --dset cifar10 \
  --mu 0.1 \
  --n_labeled 4000 \
  --batch_size 64 \
  --n_epochs 300 \
  --use_cuda