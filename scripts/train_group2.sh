#!/usr/bin/env bash

cd ..

GPU_ID=1
GPOUP_ID=2

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --group=${GPOUP_ID} \
    --num_folds=3 \
    --arch=FPMMs \
    --lr=3.5e-4 \
    --dataset='remote'

