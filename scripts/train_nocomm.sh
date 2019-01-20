#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0

# store detection results
# python -m pdb models/store_det_res.py -m sgdet -model sd_nocomm -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#         -store_det_res -save_dir data/sgdet_det_res

# python -m pdb models/train_rels.py -m sgdet -model motifnet -order confidence -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
#     -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#     -save_dir checkpoints/debug -nepoch 50 -use_bias


# python -m pdb models/train_rels.py -m sgdet -model motifnet -order confidence -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
#     -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/motifnet-conf-sgdet/vgrel-5.tar \
#     -save_dir checkpoints/debug -nepoch 50 -use_bias

# python -m pdb models/train_rels.py -m sgdet -model stanford -b 4 -p 400 -lr 1e-4 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#     -save_dir checkpoints/stanford -adam

# python -m pdb models/train_rels_nocomm.py -num_iter 1 -m sgdet -model sd_nocomm -b 6 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/sg_nocomm/sgdet/debug -nepoch 10 -use_bias -tensorboard_interval 10

## evaluation motifs
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/eval_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
#     -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/pretrained/vgrel-motifnet-sgdet.tar -nepoch 50 -cache motifnet_sgdet.pkl -use_bias \
#     -save_detection_results


########### sgdet ###################

# hw4-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_300_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 300

# hw4-2
# # SL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_300_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 300


# hw4-3
# SL
# # test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_300_1.0_MsgRmHead/iter3/sgdet-1.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 300 -debug_type try9 \
#       -test
#       # -use_postprocess \




# jun2-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9
# # # test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter2/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9 \
#       -use_postprocess \
#       -test 
# # # RL-try4
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter2/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5/iter2/sgdet-1_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
#       -use_postprocess
# # RL-try4-selfcritic
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter5/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr6_selfcritic -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 -baseline_type self_critic \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr6_selfcritic/iter5/sgdet-1_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
#       # -use_postprocess
# final_test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter5/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test -reward spice \
#       # -use_postprocess


# # jun2-1
# # SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter3/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9 \
#       -use_postprocess \
#       -test 
# # # RL-try4
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter3/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5/iter3/sgdet-7_40000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
#       -use_postprocess
# RL-try4-spice
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter3/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 -reward spice -reward_type 20 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5_spice20/iter3/sgdet-0_40000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
#       # -use_postprocess
# final_test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5/iter5/sgdet-6_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test -reward spice  \
#       # -use_postprocess


# # jun2-2
# # SL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter4/sgdet-4.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9 \
#       # -use_postprocess \
#       # -test
# # # RL-try4
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter4/sgdet-4.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5/iter4/sgdet-3_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
#       -use_postprocess
# RL-try4-constant
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter5/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr6_constant -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 -baseline_type constant \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr6_constant/iter5/sgdet-0.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
#       -use_postprocess
# final_test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5_spice20/iter5/sgdet-0.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test -reward spice \
#       # -use_postprocess


# # jun2-3
# # SL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter5/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try9 \
#       # -use_postprocess \
#       # -test 
# # # RL-try4
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter5/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5/iter5/sgdet-6_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
#       -use_postprocess

# # RL-try4-spice
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead/iter5/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 100 -debug_type try4 -reward spice -reward_type 20 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_100_1.0_MsgRmHead_rf_try4_lr5_spice20/iter5/sgdet-1_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 100 \
#       -test \
      # -use_postprocess

# dcd25-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter4/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 \
#       -use_postprocess \
#       -test 
# RL-try4
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter4/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 \
#       # -use_postprocess
# # test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5/iter4/sgdet-7_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test
#       # -use_postprocess 
# RL-try4-spice
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter4/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 -reward spice -reward_type 20 \
#       # -use_postprocess
# # test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5_spice20/iter4/sgdet-0.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test
# #       # -use_postprocess 

# dcd25-1
# SL
# RL-try4
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter4/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5/iter4/sgdet-8_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test \
#       # -use_postprocess 
# RL-try4-counterfactual
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter4/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr6_selfcritic -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 -baseline_type self_critic \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr6_selfcritic/iter4/sgdet-0.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test \
#       # -use_postprocess 


# dcd31-0
# # SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter5/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 \
#       -use_postprocess \
#       -test 
# RL-try4
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter5/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 \
#       # -use_postprocess
# # test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4/iter5/sgdet-0.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test
#       # -use_postprocess 
# RL-constant
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter4/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr6_constant -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 -baseline_type constant \
#       # -use_postprocess
# # test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr6_constant/iter4/sgdet-0.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test
#       # -use_postprocess 


# dcd31-3
# RL-try4
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter5/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4/iter5/sgdet-0.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test \
#       -use_postprocess



# dcd101-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_300_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 300
# # RL-try4
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter2/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4/iter2/sgdet-1_40000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test \
#       # -use_postprocess
# # RL-try4-spice
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter2/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 -reward spice -reward_type 20 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr5_spice20/iter2/sgdet-0_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test \
# #       # -use_postprocess
# visulation
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_0.5_MsgRmHead/iter3/sgdet-2.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 0.5 -msg_rm_head -step_obj_dim 200 \
#       -test \
#       -use_postprocess \
#       -save_detection_results 


# dcd101-1
# # SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter2/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 \
#       -use_postprocess \
#       -test
# RL-try4
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter2/sgdet-3.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4/iter2/sgdet-1_40000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test \
#       -use_postprocess 


# dcd102-1
# # SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -use_postprocess -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200
# # test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter3/sgdet-1.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 \
#       -use_postprocess \
#       -test 
# RL-try4
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 1 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead/iter3/sgdet-1.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -rec_dropout 0.0 -rl_offdropout -overlap_thresh 1.0 -msg_rm_head \
#       -step_obj_dim 200 -debug_type try4 \
#       # -use_postprocess
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgdet -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgdet/test1_200_1.0_MsgRmHead_rf_try4_lr4/iter3/sgdet-3_20000.tar \
#       -save_dir checkpoints/scenedynamic/sgdet/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -overlap_thresh 1.0 -msg_rm_head -step_obj_dim 200 \
#       -test \
#       -use_postprocess








#### sgcls debug ####

export CUDA_VISIBLE_DEVICES=3
python -m pdb models/train_rels_nocomm.py -num_iter 2 -m sgcls -model sd_nocomm -b 4 -clip 5 \
      -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
      -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
      -step_obj_dim 200









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

#################### iter3 ##########################
# jun-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter3/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter3/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100_rf_1e5/iter3/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test

# jun-1
# SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter3/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter3/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200_rf_1e5/iter3/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test

# jun-2
# SL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter3/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter3/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e5/iter3/sgcls-8.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test

################### iter4 ##############################
# dcd25-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter4/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter4/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100_rf_1e5/iter4/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# # RL-selfcritic
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_dcd101/iter5/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e6_selfcritic -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 -baseline_type self_critic \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e6_selfcritic/iter5/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test  \
#       -step_obj_dim 300 \
#       -test

# dcd25-1
# SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter4/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter4/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200_rf_1e5/iter4/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL-constant
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-6 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_dcd101/iter5/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e6_constant -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 -baseline_type constant \
# #       # -test
# # test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e6_constant/iter5/sgcls-11.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test  \
#       -step_obj_dim 300 \
#       -test

# dcd31-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter4/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter4/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e5/iter4/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL-spice
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_dcd101/iter5/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e5_spice20 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 -reward spice -reward_type 20 \
#       # -test
# # test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e5_spice20/iter5/sgcls-4.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test  \
#       -step_obj_dim 300 -reward spice \
#       -test

################### iter5 ###############################
# dcd31-3
# SL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter5/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100/iter5/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_100_rf_1e5/iter5/sgcls-0.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test

# dcd102-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter5/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200/iter5/sgcls-6.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_200_rf_1e5/iter5/sgcls-3.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test

# dcd101-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter5/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -step_obj_dim 300 -reward spice \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300/iter5/sgcls-5.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300 \
#       # -test
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m sgcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/sgcls/test1_300_rf_1e5/iter5/sgcls-7.tar \
#       -save_dir checkpoints/scenedynamic/sgcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_rl_test \
#       -step_obj_dim 300 -reward spice \
#       -test
























########## predcls ################

################################### iter2 ################################
# hw4-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter2/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter2/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100_rf_1e5/iter2/predcls-4.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test

# hw4-2
# SL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter2/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter2/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200_rf_1e5/iter2/predcls-2.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test


# hw4-3
# SL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter2/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# SL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter2/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 2 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300_rf_1e5/iter2/predcls-3.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test

################################### iter3 ###################################
# jun2-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter3/predcls-10.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter3/predcls-10.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100_rf_1e5/iter3/predcls-1.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test

# jun2-1
# SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter3/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter3/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200_rf_1e5/iter3/predcls-1.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test

# jun2-2
# SL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter3/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter3/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=2
# python -m pdb models/train_rels_nocomm.py -num_iter 3 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300_rf_1e5/iter3/predcls-2.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test

################################## iter4 #######################################
# dcd25-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter4/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter4/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100_rf_1e5/iter4/predcls-4.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test

# dcd25-1
# SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter4/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# # RL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter4/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200_rf_1e5/iter4/predcls-2.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test

# dcd31-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter4/predcls-11.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter4/predcls-11.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 4 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300_rf_1e5/iter4/predcls-0.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test

############################### iter5 #############################################

# dcd31-1
# SL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100
# test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter5/predcls-7.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100/iter5/predcls-7.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_100_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 100
# # test
# export CUDA_VISIBLE_DEVICES=3
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_100_rf_1e5/iter5/predcls-4.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 100 \
#       -test

# dcd101-0
# SL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200
# # test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter5/predcls-10.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200/iter5/predcls-10.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_200_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 200
# test
# export CUDA_VISIBLE_DEVICES=0
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_200_rf_1e5/iter5/predcls-1.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 200 \
#       -test

# dcd101-1
# SL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300 -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300
# test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter5/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test
# RL
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300/iter5/predcls-6.tar \
#       -save_dir checkpoints/scenedynamic/predcls/test1_300_rf_1e5 -nepoch 20 -use_bias -tensorboard_interval 10 \
#       -rl_train -filte_large -rec_dropout 0.0 -rl_offdropout \
#       -step_obj_dim 300
# # test
# export CUDA_VISIBLE_DEVICES=1
# python -m pdb models/train_rels_nocomm.py -num_iter 5 -m predcls -model sd_nocomm -b 4 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/scenedynamic/predcls/test1_300_rf_1e5/iter5/predcls-3.tar \
#       -save_dir checkpoints/scenedynamic/predcls/debug -nepoch 20 -use_bias -tensorboard_interval 10 -sl_train \
#       -step_obj_dim 300 \
#       -test

##########################################

