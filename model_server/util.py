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

def Dilate(x, kernel = DILATE, number_iter = 2):
    for i in range(number_iter):
        x = nn.functional.pad(x, (1, 1, 1, 1))
        x = kernel(x)
    return x

def get_dis_map(x, kernel = ERODE):
    assert x.max() <= 1 and x.min()>= 0, "tensor must in arrange 0 - 1"
    dis_map = torch.zeros_like(x)
    old_x = -1
    while old_x:
        dis_map += x
        x = nn.functional.pad(x, (1, 1, 1, 1))
        x = kernel(x)
        now_x = torch.sum(x)
        if old_x == now_x:
            return old_x, x
        old_x = now_x
    
    B,_,_,_ = dis_map.shape    
    for ind in range(B):
        dis_map[ind] = dis_map[ind]/dis_map[ind].max()

    return dis_map

# plan ours
def get_level_set( x ):
    # x = Dilate(x) 
    return get_dis_map(x)

# plan 1
def get_2D_over_mask(x, dilate_num = 1, iter_num = 3, kernel = np.ones((3, 3), np.uint8)):
    img_dilate = x
    for i in range(iter_num):
        img_dilate = cv2.dilate(img_dilate, kernel, iterations = dilate_num)
    depth_i = distance_transform_cdt(img_dilate, metric='taxicab')
    return depth_i
