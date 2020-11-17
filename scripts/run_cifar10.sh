#!/bin/bash

GPU_ID=$1
batch_size=$2
augtype=$3
n_epochs=$4

CUDA_VISIBLE_DEVICES=${GPU_ID} python ../train.py \
  --dset cifar10 \
  --mu 0.1 \
  --n_labeled 4000 \
  --batch_size ${batch_size} \
  --log_dir "../experiment/batch_size_${batch_size}" \
  --augtype ${augtype} \
  --n_epochs ${n_epochs} \
  --use_cuda