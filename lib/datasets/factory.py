# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.cityscape import cityscape
from lib.datasets.coco import coco
from lib.datasets.imagenet import imagenet
from lib.datasets.vg import vg

from lib.datasets.clipart import clipart
from lib.datasets.water import water

from lib.datasets.sim10k import sim10k
from lib.datasets.cityscape_car import cityscape_car
from lib.datasets.KITTI import kitti_car

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train_s', 'train_t', 'train_all', 'test_s', 'test_t','test_all', 'train_demo_s', 'train_demo_t', 'test_demo_s','test_demo_t']:
    name = 'cityscape_{}_{}'.format(year, split)
    devkit_path_c2f = '/media/neuron/New Volume/faster-rcnn.pytorch-pytorch-1.0/data/cityscape'
    # dataroot = '/media/neuron/New Volume/faster-rcnn.pytorch-pytorch-1.0/data'
    __sets[name] = (lambda split=split, year=year: cityscape(split, year, devkit_path_c2f))


# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))


for year in ['2007']:
  for split in ['train', 'test', 'trainval']:
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split : clipart(split,year))


for year in ['2007']:
  for split in ['train', 'test']:
    name = 'water_{}'.format(split)
    __sets[name] = (lambda split=split : water(split,year))

for split in ['train','val','train_demo', 'test_vis']:
  name = 'sim10k_{}'.format(split)
  devkit_path_sim10k = '/media/neuron/New Volume/faster-rcnn.pytorch-pytorch-1.0/data/sim10k/VOC2012'
  __sets[name] = (lambda split=split : sim10k(split, devkit_path_sim10k))

for split in ['train', 'trainval','val','test','train_demo', 'test_vis']:
  name = 'cityscape_car_{}'.format(split)
  devkit_path_ctcar = '/media/neuron/New Volume/faster-rcnn.pytorch-pytorch-1.0/data/cityscape_car/VOC2007'
  __sets[name] = (lambda split=split : cityscape_car(split, devkit_path_ctcar))

for split in ['trainval','test','train_demo', 'test_vis']:
  name = 'kitti_car_{}'.format(split)
  devkit_path_kitticar = '/media/neuron/New Volume/faster-rcnn.pytorch-pytorch-1.0/data/KITTI/VOC2012'
  __sets[name] = (lambda split=split : kitti_car(split, devkit_path_kitticar))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
