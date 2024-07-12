data='N3DV/flame_steak'
time=$(date "+%Y-%m-%d_%H-%M-%S")
logdir='debug_flame_steak'
python train.py --eval -s data/${data} --appearance_dim 0 --iterations 20 -m outputs/${logdir}/$time -r 4 --time_dim 4 --time_embedding positional_encoding --dataloader --num_workers 0 --debug