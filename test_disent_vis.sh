#!/usr/bin/env bash











# city to foggy city

# Get the visualization feature map

CUDA_VISIBLE_DEVICES=0 python test_net_da_disent_vis.py --dataset cityscape --part test_t \
    --model_dir='./model-weights/c2f_ddf_weights.pth' \
    --cuda --net vgg16 --save_name './results-c2f' \
    --vis --vis_dir './vis_heatmap/ddf-c2f'

# Get the domain distance (PAD, EMD)

#CUDA_VISIBLE_DEVICES=0 python test_net_da_disent_distance.py --dataset cityscape --part test_t \
#    --model_dir='path to your weights' \
#    --cuda --net vgg16 --save_name 'path to save the results'



# ./test_disent_vis.sh

