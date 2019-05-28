#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_dim 512 -use_bias \
      -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
      -save_dir checkpoints/debug/sgcls/debug -nepoch 100 -tensorboard_interval 10 -sl_train \
      -debug_type test1_1 -pooling_size 4

python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_dim 512 -use_bias       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet1/vg-41.tar  -save_dir checkpoints/debug/sgcls/debugW41hardnew -nepoch 100 -tensorboard_interval 10 -sl_train -debug_type test1_1 -pooling_size 5




python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_d checkpoints/vg-faster-rcnn.tar  -save_dir checkpoints/debug/sgcls/debugW2t -nepoch 100 -tensorboard_interval 10 -sl_train -debug_type test1_1 -pooling_size 5
