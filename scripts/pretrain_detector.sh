#!/usr/bin/env bash
# Train the model without COCO pretraining

# cl-train detector
# python -m pdb models/train_detector.py -b 6 -lr 1e-3 -save_dir checkpoints/vgdet -nepoch 50 -ngpu 1 -nwork 0 -p 100 -clip 5
# cl-eval detector
python -m pdb models/train_detector.py -ngpu 1 -b 6 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar -nwork 1 -p 100 -test
# 0.098, 0.204, 0.087, 0.000, 0.044, 0.122, 0.176, 0.249, 0.251, 0.000, 0.139, 0.291

# python -m pdb models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/pretrained/vg-faster-rcnn.tar -nwork 1 -p 100 -test


# python -m pdb models/train_detector.py -b 6 -lr 1e-3 -save_dir checkpoints/vgdet -nepoch 50 -ngpu 3 -nwork 3 -p 100 -clip 5

# If you want to evaluate on the frequency baseline now, run this command (replace the checkpoint with the
# best checkpoint you found).
#export CUDA_VISIBLE_DEVICES=0
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-24.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=1
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=2
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#
#
