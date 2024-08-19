python train.py --eval -s data/N3DV/flame_steak --appearance_dim 0 -m outputs/flame_steak/chunk0 -r 2 --voxel_size 0.02 --time_dim 20 --use_wandb --dataloader --num_workers 10 --time_duration 0.0 0.5 --iterations 7000 &&
python train.py --eval -s data/N3DV/flame_steak --appearance_dim 0 -m outputs/flame_steak/chunk1 -r 2 --voxel_size 0.02 --time_dim 20 --use_wandb --dataloader --num_workers 10 --time_duration 0.5 1.0 --iterations 7000 --freeze_mlp -l outputs/flame_steak/chunk0 &&
python train.py --eval -s data/N3DV/flame_steak --appearance_dim 0 -m outputs/flame_steak/chunk2 -r 2 --voxel_size 0.02 --time_dim 20 --use_wandb --dataloader --num_workers 10 --time_duration 1.0 1.5 --iterations 7000 --freeze_mlp -l outputs/flame_steak/chunk1 &&
python train.py --eval -s data/N3DV/flame_steak --appearance_dim 0 -m outputs/flame_steak/chunk3 -r 2 --voxel_size 0.02 --time_dim 20 --use_wandb --dataloader --num_workers 10 --time_duration 1.5 2.0 --iterations 7000 --freeze_mlp -l outputs/flame_steak/chunk2 &&
python train.py --eval -s data/N3DV/flame_steak --appearance_dim 0 -m outputs/flame_steak/chunk4 -r 2 --voxel_size 0.02 --time_dim 20 --use_wandb --dataloader --num_workers 10 --time_duration 2.0 2.5 --iterations 7000 --freeze_mlp -l outputs/flame_steak/chunk3 &&
python train.py --eval -s data/N3DV/flame_steak --appearance_dim 0 -m outputs/flame_steak/chunk5 -r 2 --voxel_size 0.02 --time_dim 20 --use_wandb --dataloader --num_workers 10 --time_duration 2.5 3.0 --iterations 7000 --freeze_mlp -l outputs/flame_steak/chunk4 &&
python train.py --eval -s data/N3DV/flame_steak --appearance_dim 0 -m outputs/flame_steak/chunk6 -r 2 --voxel_size 0.02 --time_dim 20 --use_wandb --dataloader --num_workers 10 --time_duration 3.0 3.5 --iterations 7000 --freeze_mlp -l outputs/flame_steak/chunk5

python render.py -m outputs/flame_steak/chunk0 --skip_train &&
python render.py -m outputs/flame_steak/chunk1 --skip_train &&
python render.py -m outputs/flame_steak/chunk2 --skip_train &&
python render.py -m outputs/flame_steak/chunk3 --skip_train &&
python render.py -m outputs/flame_steak/chunk4 --skip_train &&
python render.py -m outputs/flame_steak/chunk5 --skip_train &&
python render.py -m outputs/flame_steak/chunk6 --skip_train
