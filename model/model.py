import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util import MinPool
from .RESUNet import ResBlock

class RCS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, strides = 1, padding = 1, block_number = 2) -> None:
        super().__init__()
        d = [
                ResBlock(in_channels, out_channels, strides)
            ]
   
        for i in range(block_number - 1):
            d.append(  ResBlock(out_channels, out_channels, strides) )
        self.body = nn.Sequential(*d)

    def forward(self, x):
        x = self.body(x)
        return x

class DCBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, strides = 2):
        super(DCBL, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size = kernel_size,
                                        stride = strides, bias = True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU() 

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out

class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() 

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        out = self.relu(x1)
        x2 = self.conv2(out)
        x2 = self.bn2(x2)
        out = self.relu(x2)
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() 

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out


class Unet(nn.Module):
    def __init__(self, need_return_dict = True):
        super(Unet, self).__init__()
        self.need_return_dict = need_return_dict
        self.layer1_conv = double_conv2d_bn(1, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        
        self.layer10_conv = nn.Conv2d(8, 1, kernel_size = 3,
                                      stride = 1, padding = 1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        self.sigmoid = nn.Sigmoid()
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)

    def build_result(self, x, y):
        return {
            "mask": x,
            "edge": y,
        }

    def forward(self, x):
        # print(x.shape)
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        edge = nn.functional.pad(outp, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return  self.build_result(outp, edge) if self.need_return_dict else (outp, edge) 
    

class RUnet(nn.Module):
    def __init__(self, need_return_dict = True):
        super(RUnet, self).__init__()
        self.need_return_dict = need_return_dict
        self.layer1_conv = RCS(1, 8)
        self.layer2_conv = RCS(8, 16)
        self.layer3_conv = RCS(16, 32)
        self.layer4_conv = RCS(32, 64)
        self.layer5_conv = RCS(64, 128)
        self.layer6_conv = RCS(128, 64)
        self.layer7_conv = RCS(64, 32)
        self.layer8_conv = RCS(32, 16)
        self.layer9_conv = RCS(16, 8)
        self.layer10_conv = nn.Conv2d(8, 1, kernel_size = 3,
                                      stride = 1, padding = 1, bias=True)

        self.deconv1 = DCBL(128, 64)
        self.deconv2 = DCBL(64, 32)
        self.deconv3 = DCBL(32, 16)
        self.deconv4 = DCBL(16, 8)

        self.sigmoid = nn.Sigmoid()
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)

    def build_result(self, x, y):
        return {
            "mask": x,
            "edge": y,
        }

    def forward(self, x):
        # print(x.shape)
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        edge = nn.functional.pad(outp, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return  self.build_result(outp, edge) if self.need_return_dict else (outp, edge) 
