#!/usr/bin/env bash


# city to foggy city
CUDA_VISIBLE_DEVICES=0 python test_net_da_disent.py --dataset cityscape --part test_t \
    --model_dir='./model-weights/c2f_ddf_weights.pth' \
    --cuda --net vgg16 --save_name './results-preds/c2f-ddf'


# sim10k to city, car only


#CUDA_VISIBLE_DEVICES=0 python test_net_da_disent.py --dataset sim10k2cityscape --part test_t \
#    --model_dir='./model-weights/s2c_ddf.pth' \
#    --cuda --net vgg16 --save_name './results-preds/s2c-ddf'

# kitti to city, car only

#CUDA_VISIBLE_DEVICES=0 python test_net_da_disent.py --dataset kitti2cityscape --part test_t \
#    --model_dir='./model-weights/k2c_ddf.pth' \
#    --cuda --net vgg16 --save_name './results-preds/k2c-ddf'

# city to kitti, car only

#CUDA_VISIBLE_DEVICES=0 python test_net_da_disent.py --dataset cityscape2kitti --part test_t \
#    --model_dir='./model-weights/c2k_ddf.pth' \
#    --cuda --net vgg16 --save_name './results-preds/c2k-ddf'

# ./test_disent.sh
