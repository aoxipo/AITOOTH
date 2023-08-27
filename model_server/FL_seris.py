


import torch
import torch.nn as nn
import torch.nn.functional as F

from .RESUNet import ResBlock
from .util import MinPool
from .model import RCS, DCBL, RC

class DecodeBlock(nn.Module):
    def __init__(self,in_channel, middle_channel = [8, 16, 32, 64, 128], block_number = 2):
        super().__init__( )
        self.pre = nn.Conv2d(in_channel, middle_channel[0], 1, 1)
        
        self.encode_1 = Encode(middle_channel[0], middle_channel[1], block_number)
        self.encode_2 = Encode(middle_channel[1], middle_channel[2], block_number)
        self.encode_3 = Encode(middle_channel[2], middle_channel[3], block_number)
        self.encode_4 = Encode(middle_channel[3], middle_channel[4], block_number)
        
    def forward(self, x):
        x = self.pre(x)
        x1 = self.encode_1(x)
        x2 = self.encode_2(x1)
        x3 = self.encode_3(x2)
        x4 = self.encode_4(x3)
        return x1, x2, x3, x4

class Encode(nn.Module):
    def __init__(self, in_channel, out_channel, block_number = 1, conv_type = "conv"):
        super().__init__( )
        if conv_type == "conv":
            self.conv = RC(in_channel, out_channel, block_number = block_number)
        else:
            self.conv = RCS(in_channel, out_channel, block_number = block_number)
        self.downsample = nn.MaxPool2d(2)
        
    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.downsample(x_conv)
        return x_conv, x_pool

class EncodeBlock(nn.Module):
    def __init__(self, in_channel, block_number = [ 2, 2, 2, 2], middle_channel = [8, 16, 32, 64, 128], conv_type = "conv"):
        super().__init__( )
        self.encode_0 = Encode(in_channel, middle_channel[0], block_number[0], conv_type)
        self.encode_1 = Encode(middle_channel[0], middle_channel[1], block_number[0], conv_type)
        self.encode_2 = Encode(middle_channel[1], middle_channel[2], block_number[1], conv_type)
        self.encode_3 = Encode(middle_channel[2], middle_channel[3], block_number[2], conv_type)
        self.encode_4 = Encode(middle_channel[3], middle_channel[4], block_number[3], conv_type)
        
        
    def forward(self, x):
        x0_conv, x0_pool = self.encode_0(x)
        x1_conv, x1_pool = self.encode_1(x0_pool)
        x2_conv, x2_pool = self.encode_2(x1_pool)
        x3_conv, x3_pool = self.encode_3(x2_pool)
        x4_conv, x4_pool = self.encode_4(x3_pool)
        return x0_conv, x1_conv, x2_conv, x3_conv, x4_conv

class Decode(nn.Module):
    def __init__(self, in_channel, out_channel, conv_type = "conv"):
        super().__init__( )
        self.deconv = DCBL( in_channel, out_channel)
        self.conv = RCS(in_channel, out_channel, conv_type)
        
    def forward(self, x, y):
        x = self.deconv(x)
        # print(x.shape, y.shape)
        concat = torch.cat([x, y], dim=1)
        x = self.conv(concat)
        return x

class UBlock(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 16, middle_channel = [ 8, 16, 32, 64, 128 ]):
        super().__init__()
        self.encode = EncodeBlock(in_channel, block_number = [2, 2, 2, 2], middle_channel = middle_channel , conv_type = "conv")
        self.up = nn.Upsample(scale_factor = 2)
        self.decode_0 = Decode(middle_channel[1], middle_channel[0], conv_type = "conv")
        self.decode_1 = Decode(middle_channel[2], middle_channel[1], conv_type = "conv")
        self.decode_2 = Decode(middle_channel[3], middle_channel[2], conv_type = "conv")
        self.decode_3 = Decode(middle_channel[4], middle_channel[3], conv_type = "conv")
        self.final = nn.Conv2d( middle_channel[0], out_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encode(x)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        # x4 = self.up(x4)

        x_1 = self.decode_3(x4, x3)
        
        x_2 = self.decode_2(x_1, x2)
        
        x_3 = self.decode_1(x_2, x1)
        
        x_4 = self.decode_0(x_3, x0)
        # print(x_4.shape)
        outp = self.sigmoid(self.final(x_4))
        return  outp, x_4
from .model import Unet
from .FL_base import FL_base
class FL_tiny(FL_base):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1,
                embed_shape = ( 2, 4),
                batch_size = 16,
                need_return_dict = False
        ):
        super(FL_tiny, self).__init__()
     
        self.batch_size = batch_size
        self.embed_shape = embed_shape
        self.need_return_dict = need_return_dict
        self.middle_channel = middle_channel
       
        # replace your model
        ####################################
        self.model = nn.Sequential(
            UBlock(middle_channel, middle_channel)
        )
        ####################################
        self.edge_body = nn.Sequential(
            ResBlock(8,4),
            nn.Conv2d( 4, out_channel, 1, 1),
            nn.ReLU(),
        )
        ####################################
        self.consit_body = nn.Sequential(
            nn.Conv2d( middle_channel, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),

            nn.Conv2d( 32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d( 64, 8, 2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
        )
        ####################################
        
        self.final = nn.Conv2d(8, out_channel, 1,1 )
        self.edge_final = nn.Conv2d(8, out_channel, 1,1 )
        self.upsample = nn.Upsample(scale_factor = 2)
        
    def ext_feature(self, x):
        B,C,W,H =  x.shape
        x_embed = self.get_embeding(x) 
        batch_item_combined_hm_preds = []
        for batch_index in range(B): 

            batch_item_x_embed = x_embed[batch_index,:,:,:,:]
            # print(batch_item_x_embed.shape)
            # assert False, (batch_item_x_embed.shape)
            #### your forward model here
            output, _ = self.model( batch_item_x_embed ) # only for mask not edge, edge will have another way
            #### 
                
            batch_item_combined_hm_preds.append(output)
            
        x_combine = torch.stack(batch_item_combined_hm_preds, 0)
        outp = self.re_build(x_combine)
        
        return outp
    
    def consist(self, x):
        x = x.clone().detach()
        x = self.consit_body(x)
        return x
    
    def edge_hot_map(self, x):
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        edge = self.edge_body(edge)
        return edge
    
    def forward(self, x):   
        x = self.downsample(x)
        # print(x.shape)
        x = self.ext_feature(x)

        outp = self.consist(x)
        # assert False, (x.shape, outp.shape)
        edge = self.edge_hot_map(outp)
        
        outp = self.final(outp)
        outp = self.sigmod(outp)
        # print(outp.shape, edge.shape)
        # outp = self.upsample(outp)
        # x = self.upsample(x)
        # assert False, (outp.shape, self.upsample(x).shape, edge.shape)
        return self.build_results(x, outp) if (self.need_return_dict) else ( x, outp, edge)

