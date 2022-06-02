import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


class DSEncoder(nn.Module):
    def __init__(self,dim=512):
        super(DSEncoder, self).__init__()


        self.enc_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        # self.relu1 = nn.ReLU(inplace=True),

        # nn.Conv2d(256, 256, 4, 2, 1),
        # nn.ReLU(inplace=True),
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.enc_conv2 = nn.Conv2d(256, 512, 3, 1, 1)
        # self.relu2 = nn.ReLU(inplace=True),

        self.maxpool2 = nn.MaxPool2d(2, 2)
        # nn.Conv2d(512, 512, 4, 2, 1),
        # nn.ReLU(inplace=True),

        self.enc_conv3 = nn.Conv2d(512, dim, 3, 1, 1)
        # self.relu3 = nn.ReLU(inplace=True)

        # self.tanh = nn.Tanh()

    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.enc_conv1, 0, 0.01)
      normal_init(self.enc_conv2, 0, 0.01)
      normal_init(self.enc_conv3, 0, 0.01)

    def forward(self, input):

        out = F.relu(self.enc_conv1(input))
        out = self.maxpool1(out)

        out = F.relu(self.enc_conv2(out))
        out = self.maxpool2(out)

        out = F.relu(self.enc_conv3(out))

        return out





def CORAL_loss(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss


def diff_loss(invarient, specific):


    multp = invarient.t() @ specific

    # frobenius norm between source and target
    loss = torch.mean(torch.norm(multp, p="fro").pow(2))

    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss