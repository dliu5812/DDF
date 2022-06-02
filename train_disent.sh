#!/usr/bin/env bash

# ./train_disent.sh


## cityscape to foggy cityscape, all classes

CUDA_VISIBLE_DEVICES=0 python da_trainval_net_disent.py --dataset cityscape \
    --net vgg16 --bs 1 --lr 1e-3 --lr_decay_step 10 --cuda \
    --save_dir './results-models/models-ddf-vgg16-demo' \
    --epochs 14 --iterations_per_epoch 5000


# sim10k to cityscape, car
#CUDA_VISIBLE_DEVICES=0 python da_trainval_net_disent.py --dataset sim10k2cityscape \
#    --net vgg16 --bs 1 --lr 1e-3 --lr_decay_step 10 --cuda \
#    --save_dir './results-models/models-ddf-vgg16-demo' \
#    --epochs 14 --iterations_per_epoch 5000



# kitti to cityscape, car
#CUDA_VISIBLE_DEVICES=0 python da_trainval_net_disent.py --dataset kitti2cityscape \
#    --net vgg16 --bs 1 --lr 1e-3 --lr_decay_step 10 --cuda \
#    --save_dir './results-models/models-ddf-vgg16-demo' \
#    --epochs 14 --iterations_per_epoch 5000


# cityscape to kitti, car
#CUDA_VISIBLE_DEVICES=0 python da_trainval_net_disent.py --dataset cityscape2kitti \
#    --net vgg16 --bs 1 --lr 1e-3 --lr_decay_step 10 --cuda \
#    --save_dir './results-models/models-ddf-vgg16-demo' \
#    --epochs 14 --iterations_per_epoch 5000
