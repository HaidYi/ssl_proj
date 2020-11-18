#!/bin/bash

dset=$1
n_class=$2
GPU_ID=$3
batch_size=$4
augtype=$5
n_epochs=$6

CUDA_VISIBLE_DEVICES=${GPU_ID} python ../train.py \
  --dset ${dset} \
  --n_class ${n_class} \
  --mu 0.1 \
  --n_labeled 4000 \
  --batch_size ${batch_size} \
  --log_dir "../experiment/batch_size_${batch_size}" \
  --augtype ${augtype} \
  --n_epochs ${n_epochs} \
  --use_cuda