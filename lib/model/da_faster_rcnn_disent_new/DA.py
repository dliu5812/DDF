from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
from lib.model.da_faster_rcnn_disent_new.LabelResizeLayer import ImageLabelResizeLayer
from lib.model.da_faster_rcnn_disent_new.LabelResizeLayer import InstanceLabelResizeLayer



class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


class GRWLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs * ctx.alpha
        return output



class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)
        return x,label




class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(4096, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()


    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        label = self.LabelResizeLayer(x, need_backprop)
        return x,label


class _InsDAFC(nn.Module):
    def __init__(self):
        super(_InsDAFC, self).__init__()
        self.dc_ip1 = nn.Linear(4096, 1024)
        self.dc_relu1 = nn.ReLU(inplace=False)
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU(inplace=False)
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = F.sigmoid(self.clssifer(x))
        return x


class _ImageDADeep(nn.Module):
    def __init__(self, dim):
        super(_ImageDADeep, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv5 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)
        # self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self, x):
        # x=grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.reLu(self.Conv2(x))
        x = self.reLu(self.Conv3(x))
        x = self.reLu(self.Conv4(x))
        x = self.reLu(self.Conv5(x))
        # label=self.LabelResizeLayer(x,need_backprop)
        return x





class netD_ins_bi_conv(nn.Module):
    def __init__(self, feat_d, num_classes = 2):
        super(netD_ins_bi_conv, self).__init__()
        self.fc1 = nn.Linear(feat_d,100) # 1024
        self.bn1 = nn.BatchNorm1d(100)  # 1024
        self.fc2 = nn.Linear(100, 100)   # 1024
        self.bn2 = nn.BatchNorm1d(100)   # 1024
        self.fc3 = nn.Linear(100,num_classes)  # 1024
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        out = self.fc3(x)
        return out  #[256, 2]


class _InstanceDAROI(nn.Module):
    def __init__(self):
        super(_InstanceDAROI,self).__init__()

        self.dc_ip0 = nn.Linear(512 * 7 * 7, 4096)
        self.dc_relu0 = nn.ReLU()
        self.dc_drop0 = nn.Dropout(p=0.5)

        self.dc_ip1 = nn.Linear(4096, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 256)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(256,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()


    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x = x.view(x.size(0), -1)
        # print('ins da inp shape', x.size())
        x = self.dc_drop0(self.dc_relu0(self.dc_ip0(x)))
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        label = self.LabelResizeLayer(x, need_backprop)
        return x,label





class TripletLossMean(nn.Module):
    def __init__(self):
        super(TripletLossMean, self).__init__()
        # self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).mean()

    def forward(self, anchor, positive, negative, margin = 1.0):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)


        tp_loss_raw = distance_positive - distance_negative + margin

        epsilon=1e-6
        losses = tp_loss_raw.clamp(min=epsilon).mean()

        return losses