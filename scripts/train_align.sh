#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_dim 512 -use_bias \
      -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
      -save_dir checkpoints/debug/sgcls/debug -nepoch 100 -tensorboard_interval 10 -sl_train \
      -debug_type test1_1 -pooling_size 4

python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_dim 512 -use_bias \     
-p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet1/vg-41.tar \ 
-save_dir checkpoints/debug/sgcls/debugW41hardnew -nepoch 100 -tensorboard_interval 10 -sl_train \
-debug_type test1_1 -pooling_size 5



python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_d checkpoints/vg-faster-rcnn.tar \ 
-save_dir checkpoints/debug/sgcls/debugW2t -nepoch 100 -tensorboard_interval 10 -sl_train \
-debug_type test1_1 -pooling_size 5

###Resnet
python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_dim 1024 -use_bias \
      -p 100 -pooling_dim 2048 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdetff/vg-11.tar \
      -save_dir checkpoints/debug/sgcls/debug -nepoch 100 -tensorboard_interval 10 -sl_train \
      -debug_type test1_1 -pooling_size 5
      
      
########## detector
export CUDA_VISIBLE_DEVICES=0
python -m pdb models/train_detector.py -b 6 -lr 1e-3 -p 100  -clip 5 -ngpu 1 -nwork 3 -save_dir checkpoints/pretrained/vgdet 




 python tools/train_net_step_rel.py --dataset oit_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5.yaml --nw 8 --use_tfboard
     
