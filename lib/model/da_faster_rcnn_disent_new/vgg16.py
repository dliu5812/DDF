# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from lib.model.da_faster_rcnn_disent_new.faster_rcnn import _fasterRCNN
from lib.model.da_faster_rcnn_disent_new.da_disentengle import DSEncoder
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.img_size = (150, 300)    # cityscape： 150, 300

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # res = models.resnet50()
    # print(res.features)
    # print(vgg.features)
    # not using the last maxpool layer
    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])

    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:-1])

    self.ds_enc_s = DSEncoder()
    # self.decoder = Decoder(size=self.img_size)

    # self.ds_enc_s = DSEncoderDeep()
    # self.decoder = DecoderDeep(size=self.img_size)

    # self.ins_roi_rs = nn.Linear(4096, self.dout_base_model)

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)



    # print('n class', self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

  # def ins_roi_feat_rs(self, feat_inp, out_dim = 256):
  #
  #   in_dim = feat_inp.size(-1)
  #   feat_out = nn.Linear(in_dim, out_dim)



