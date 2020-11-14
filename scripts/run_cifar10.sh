#!/bin/bash

GPU_ID=$1
batch_size=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python ../train.py \
  --dset cifar10 \
  --mu 0.1 \
  --n_labeled 250 \
  --batch_size ${batch_size} \
  --log_dir "../experiment/batch_size_${batch_size}" \
  --n_epochs 1 \
  --use_cuda