export CUDA_VISIBLE_DEVICES=1
python -m pdb models/eval-rel.py -m sgcls -model align -b 1 -clip 5 -use_bias \
      -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/motifnet-conf-sgcls/vgrel-11.tar \
      -nepoch 50 -tensorboard_interval 10 -test -cache baseline_sgcls \
      
      
# python -m pdb models/eval-rel.py -m sgcls -model align -b 1 -clip 5 -use_bias \
     # -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/debug/sgcls/debug2/sgcls-16 \
     # -nepoch 50 -tensorboard_interval 10 -test -cache baseline_predcls \
          
