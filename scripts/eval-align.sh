export CUDA_VISIBLE_DEVICES=1
python -m pdb models/eval-rel.py -m sgcls -model align -b 1 -clip 5 -use_bias \
      -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/debug/sgcls/debug2/sgcls-16 \
      -nepoch 50 -tensorboard_interval 10 -test -cache baseline_sgcls \
      
      
# python -m pdb models/eval-rel.py -m sgcls -model align -b 1 -clip 5 -use_bias \
     # -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/debug/sgcls/debug2/sgcls-16 \
     # -nepoch 50 -tensorboard_interval 10 -test -cache baseline_predcls \
     
 python -m pdb models/train_align.py -m sgcls -model align -b 1 -clip 5 -hidden_dim 512 -use_bias \ 
 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/debug/sgcls/debugHKSelumean/sgcls-9.tar \
 -save_dir checkpoints/debug/sgcls/debugW41hardnew -nepoch 100 -tensorboard_interval 10 -sl_rl_test \ 
 -debug_type test1_1 -pooling_size 5 -test -cache caches/Align_sgcls.pkl
          
