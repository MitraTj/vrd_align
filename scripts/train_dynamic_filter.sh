#!/usr/bin/env bash

# # dcd25
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 4 -clip 5 \
#       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/debug/sgcls/test6_2_spatial -nepoch 100 -tensorboard_interval 10 -sl_train \
#       -debug_type test6_2_spatial -num_iter 2 -hidden_dim 512



# dcd25
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 4 -clip 5 \
#       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/debug/sgcls/test5_8_bn -nepoch 100 -tensorboard_interval 10 -sl_train \
#       -debug_type test5_8_bn


# # ntu186
# export CUDA_VISIBLE_DEVICES=0
# ################ batch size = 3 ################
# python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 3 -clip 5 \
#       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/debug/sgcls/test6_3 -nepoch 100 -tensorboard_interval 10 -sl_train \
#       -debug_type test6_3 -num_iter 2 -hidden_dim 512



# # # # ntu186
# export CUDA_VISIBLE_DEVICES=1
# ################### batch size = 3 ###############
# python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 3 -clip 5 \
#       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/debug/sgcls/test6_4 -nepoch 100 -tensorboard_interval 10 -sl_train \
#       -debug_type test6_4 -num_iter 2 -hidden_dim 512


# # # # ntu186
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 4 -clip 5 \
#       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/debug/sgcls/test6_2 -nepoch 100 -tensorboard_interval 10 -sl_train \
#       -debug_type test6_2 -num_iter 2 -hidden_dim 512




# # ntu186
# export CUDA_VISIBLE_DEVICES=3
# ################### batch size = 3 ################
# python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 3 -clip 5 \
#       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/debug/sgcls/test6_5 -nepoch 100 -tensorboard_interval 10 -sl_train \
#       -debug_type test6_5 -num_iter 2 -hidden_dim 512




# # dcd31
# export CUDA_VISIBLE_DEVICES=0
# ############# batch_size = 3 #########
# python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 3 -clip 5 \
#       -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/debug/sgcls/test6_2_spatial_b3 -nepoch 100 -tensorboard_interval 10 -sl_train \
#       -debug_type test6_2_spatial -num_iter 2 -hidden_dim 512


# # dcd31
export CUDA_VISIBLE_DEVICES=1
################# batch size = 3 ####################
python -m pdb models/train_dynamic_filter.py -m sgcls -model dynamic_filter -b 3 -clip 5 \
      -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
      -save_dir checkpoints/debug/sgcls/test6_6 -nepoch 100 -tensorboard_interval 10 -sl_train \
      -debug_type test6_6








############# sgcls ##########
# # sgcls sl train
##################### iter2 ###################
# hw4-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter2/sgcls-8.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL-recall
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter2/sgcls-8.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100_rf_1e5/iter2/sgcls-2.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL-SPICE
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter2/sgcls-8.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100_rf_1e5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100 -reward spice -reward_type 20 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100_rf_1e5_spice/iter2/sgcls-0.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -step_obj_dim 100 \
#       -test
# RL-constant
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter2/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200_rf_1e6_constant -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200  -baseline_type constant \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200_rf_1e6_constant/iter2/sgcls-1.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -step_obj_dim 200 \
#       -test



# hw4-2
# SL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter2/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter2/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200_rf_1e5/iter2/sgcls-4.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL-spice
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter2/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200_rf_1e5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200 -reward spice -reward_type 20 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200_rf_1e5_spice/iter2/sgcls-0.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -step_obj_dim 200 \
#       -test
# RL-self_critic
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter2/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200_rf_1e6_selfcritic -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200  -baseline_type self_critic \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200_rf_1e6_selfcritic/iter2/sgcls-0.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -step_obj_dim 200 \
#       -test


# hw4-3
# SL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter2/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter2/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e5/iter2/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL-spice
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter2/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 -reward spice -reward_type 20 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e5_spice/iter2/sgcls-0.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -step_obj_dim 300 \
#       # -test





