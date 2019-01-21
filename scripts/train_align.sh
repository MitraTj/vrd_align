#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=2
python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_dim 512 -use_bias \
      -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
      -save_dir checkpoints/debug/sgcls/debug -nepoch 100 -tensorboard_interval 10 -sl_train \
      -debug_type test1_0



