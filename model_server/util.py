from torch import nn
import torch
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_cdt
import numpy as np
import cv2
import copy
Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class DWConv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(DWConv, self).__init__()
        self.inp_dim = inp_dim
        self.depthwise = nn.Conv2d(inp_dim, inp_dim, kernel_size, stride, padding=(kernel_size-1)//2, groups=inp_dim,bias=inp_dim)
        self.pointwise = nn.Conv2d(inp_dim, out_dim, kernel_size=1, groups=1)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class depthwise_separable_conv(nn.Module):
    def init(self, nin, nout, kernel_size=3, stride = 1, bn = False, relu = True):
        super(depthwise_separable_conv, self).init()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, Conv_method = Conv):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        middle_dim = 1 if int(out_dim/2) <= 0 else int(out_dim/2)
        self.conv1 = Conv(inp_dim, middle_dim, 1, relu=False)
        self.bn2 = nn.BatchNorm2d(middle_dim)
        self.conv2 = Conv_method(middle_dim, middle_dim, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(middle_dim)
        self.conv3 = Conv(middle_dim, out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Sequential(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1),
    nn.BatchNorm2d(out_channels))

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

class MinPool(nn.Module):
    def __init__(self, kernel_size, ndim=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool, self).__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                                  return_indices=return_indices, ceil_mode=ceil_mode)
    def forward(self, x):
        x = self.pool(-x)
        return -x

@torch.no_grad()
class OpenOpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.erode = MinPool(2, 2, 1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
    def forward(self, x):
        x = nn.functional.pad(x, (1, 0, 1, 0))
        x = self.erode(x)
        # x = nn.functional.pad(x, (1, 0, 1, 0))
        # x = self.dilate(x) 
        # assert False, x.shape
        return x

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

def crop_tensor(image_pack, scale_x, scale_y = None):
    if scale_y is None:
        scale_y = scale_x
    _, _, w, h = image_pack.size()
    a = int(w/scale_x)
    b = int(h/scale_y)
    # print(a, b)
    t = torch.split(image_pack, a, dim = 2)
    ans = []
    for i in t:
        for j in torch.split(i, b, dim=3):
            ans.append(j)
            # print(j.shape)
    d = torch.stack(ans, 1)
    return d

def cat_tensor(image_pack, scale_x, scale_y = None):
    if scale_y is None:
        scale_y = scale_x
    data = []
    for i in range(scale_x):
        m = []
        for j in range(scale_y):
            m.append(image_pack[:, i * scale_y + j ,:,:,:])
            # print(  i * scale_y + j, i,j )
        data.append(torch.cat(m, dim = -1))
    
    data = torch.cat(data, dim = -2)
    return data

ERODE = MinPool(3,2,1).cuda()
DILATE = nn.MaxPool2d(3, stride = 1).cuda()

def Dilate(x, kernel = DILATE, number_iter = 4):
    for i in range(number_iter):
        x = nn.functional.pad(x, (1, 1, 1, 1))
        x = kernel(x)
    return x

def get_dis_map(x, kernel = ERODE):
    dis_map = torch.zeros_like(x)
    while torch.sum(x):
        dis_map += x
        x = nn.functional.pad(x, (1, 1, 1, 1))
        x = kernel(x)
    return dis_map

# plan ours
def get_level_set( x ):
    x = Dilate(x)
    return get_dis_map(x)

# plan 1
def get_2D_over_mask(x, dilate_num = 1, iter_num = 3, kernel = np.ones((3, 3), np.uint8)):
    img_dilate = x
    for i in range(iter_num):
        img_dilate = cv2.dilate(img_dilate, kernel, iterations = dilate_num)
    depth_i = distance_transform_cdt(img_dilate, metric='taxicab')
    return depth_i
