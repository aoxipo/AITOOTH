import torch
import torch.nn as nn
from .RESUNet import MinPool, ResBlock
from .model import Unet
from .FL_seris import Encode, Decode

class FPN(nn.Module):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                # block_layers=[6, 12, 24, 16], 
                # transition_layer = [256, 512, 1024, 1024],
                middle_channel = [16, 32, 64, 128],
                need_return_dict = False
        ):
        super(FPN,self).__init__()
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2,2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        self.pre_encode = nn.Sequential(
            Encode(in_channel, middle_channel[0], 4)
        )
        self.out = nn.Sequential(
            nn.Conv2d( middle_channel[0], out_channel,1,1)
        )
        self.last_decode = Decode(middle_channel[1], middle_channel[0], conv_type = "conv") 
        self.encode = nn.ModuleList(
            [ Encode(
                2 * middle_channel[i], 
                middle_channel[ i+1 ], 2, 
                conv_type = "conv"
            )  for i in range(3) ]
        )
        self.decode = nn.ModuleList(
            [
                Decode(
                    2 * middle_channel[3 - i], 
                    2 * middle_channel[3 - i - 1], 
                    conv_type = "resconv") 
                for i in range(3)
            ]
        )
        
        
        self.CBL1 = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel[0],3, 1, 1),
            nn.BatchNorm2d(middle_channel[0]),
            nn.ReLU(),
        )
        
        self.CBL2 = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel[1], 3, 1, 1),
            nn.BatchNorm2d(middle_channel[1]),
            nn.ReLU(),
        )
        
        self.CBL3 = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel[2], 3, 1, 1),
            nn.BatchNorm2d(middle_channel[2]),
            nn.ReLU(),
        )
        
        self.CBL4 = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel[3], 1, 1),
            nn.BatchNorm2d(middle_channel[3]),
            nn.ReLU(),
        )
   
    def build_feature_pyramid(self, x): # 80
        x_1024 = self.downsample(x)     # 40
        x_512 = self.downsample(x_1024) # 20
        x_256 = self.downsample(x_512)  # 10
        x_128 = self.downsample(x_256)  # 5
        # print( x_1024.shape, x_512.shape, x_256.shape, x_128.shape )
        return   x, x_1024, x_512, x_256, x_128

    def feature(self, x):
        x_1 = self.CBL1(x[1])
        x_2 = self.CBL2(x[2])
        x_3 = self.CBL3(x[3])
        x_4 = self.CBL4(x[4])
        
        xc_0, xp_0 = self.pre_encode(x[0])
        
        x_c_0 = torch.cat([xp_0, x_1], 1)
        ec_0, ep_0 = self.encode[0](x_c_0)
        
        x_c_1 = torch.cat([ep_0, x_2], 1)
        ec_1, ep_1 = self.encode[1](x_c_1)
        
        x_c_2 = torch.cat([ep_1, x_3], 1)
        ec_2, ep_2 = self.encode[2](x_c_2)
        
        x_c_3 = torch.cat([ep_2, x_4], 1)
        # print(xc_0.shape, ec_0.shape, ec_1.shape, ec_2.shape, x_c_3.shape)
        
        d_0 = self.decode[0](x_c_3, ec_2)
        d_1 = self.decode[1](d_0, ec_1)
        d_2 = self.decode[2](d_1, ec_0)
        d_3 = self.last_decode(d_2, xc_0)
        out = self.out(d_3)
        # print(out.shape)
        edge = nn.functional.pad(out, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return  out, edge
    
    
    def forward(self, x):
        x= self.feature(self.build_feature_pyramid(x))
        return x

from .util import cat_tensor, crop_tensor

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, padding = 1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = (3,3), stride = (1,1) , padding = (1,1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size = (3,3), stride = (1,1), padding = (1,1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = nn.Conv2d( in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        x = inputs.clone().detach()
        out = self.layer(x)
        residual  = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)

class Hourglass2D(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass2D, self).__init__()
        nf = f + increase
        self.up1 = ResBlock2D(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d((2,2))
        self.low1 = ResBlock2D(f, nf)
        self.n = n
        # Recursive hourglass2d
        if self.n > 1:
            self.low2 = Hourglass2D(n-1, nf, bn=bn)
        else:
            self.low2 = ResBlock2D(nf, nf)
        self.low3 = ResBlock2D(nf, f)
        self.up2 = nn.Upsample(scale_factor = (2,2), mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

class DShotConnect2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_r = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1) ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_l = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1) ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(2 * out_channels),
            nn.Conv2d( 2 * out_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1) ),
            nn.ReLU(),
        )
  
    def forward(self, inputs):
        x_r = self.conv_r(inputs)
        x_l = self.conv_l(inputs)
        x = torch.cat([x_r, x_l], dim = 1)
        x = self.conv(x)
        return x

from .RESUnetGAN import Generator 
class FL2D(nn.Module):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                middle_channel = 32, 
                embed_shape = ( 2, 4),
                nstack = 2,
                batch_size = 16,
                need_return_dict = False
        ):
        super(FL2D, self).__init__()
        self.nstack = nstack
        self.batch_size = batch_size
        self.embed_shape = embed_shape
        self.need_return_dict = need_return_dict
        
        self.downsample = nn.Sequential(
            # nn.Identity(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channel, middle_channel, 1, 1)
        )
        
        self.upsample = nn.Upsample(scale_factor=2) # nn.Identity() # nn.Upsample(scale_factor=2) #nn.ConvTranspose2d(out_channel, out_channel, (2,2), (2,2))
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        
        self.hgs = nn.ModuleList( [
            nn.Sequential(
                # Hourglass2D(4, middle_channel, increase = 32)
                Generator( middle_channel, middle_channel, middle_channel = [32, 64, 128, 256] )
            ) for i in range(nstack)] 
        )
        
        self.features = nn.ModuleList( [
                nn.Sequential(
                    ResBlock2D(middle_channel, middle_channel),
                    DShotConnect2D(middle_channel, middle_channel),
                ) for i in range(nstack)
        ])
        
        self.outs = nn.ModuleList( [
            DShotConnect2D(middle_channel, middle_channel)  for i in range(nstack)
        ])
        
        self.merge_features = nn.ModuleList( [
                DShotConnect2D(middle_channel, middle_channel)  for i in range(nstack - 1)
        ] )
        self.merge_preds = nn.ModuleList( [ DShotConnect2D(middle_channel, middle_channel) for i in range(nstack - 1)] )
        self.final = nn.ModuleList( [ nn.Conv2d(nstack * middle_channel, out_channel, kernel_size = 1, stride = 1 ) for i in range(self.batch_size)] )
        # self.final = nn.Conv2d(nstack * middle_channel, out_channel, kernel_size = (3,3), stride = (1,1), padding = (1,1) )
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        
        
    
    def get_embeding(self, x):
        embed_x = crop_tensor(x, self.embed_shape[0], self.embed_shape[1])
        # embed_x = embed_x.permute(0, 2, 1, 3, 4)
        return embed_x
    
    def re_build(self, x):
        # x = x.permute(0, 2, 1, 3, 4)
        x = cat_tensor(x, self.embed_shape[0], self.embed_shape[1])        
        return x

    def build_results(self,x,y):
        return {
            "mask":x,
            'edge':y,
        }
    def forward(self, x):   
        x = self.downsample(x)
        B,C,W,H =  x.shape
        x_embed = self.get_embeding(x) 
        batch_item_combined_hm_preds = []
        
        for batch_index in range(B): 
            batch_item_x_embed = x_embed[batch_index,:,:,:,:]
            combined_hm_preds = []
            # print("batch_item_x_embed:", batch_item_x_embed.shape)
            
            for i in range(self.nstack):
                hg = self.hgs[i](batch_item_x_embed)
                # print("hg:",hg.size()) 
                feature = self.features[i](hg)
                # print("feature:",feature.size())
                preds = self.outs[i](feature)
                keys = self.sigmod(preds)
                # print("preds:", preds.size())
                combined_hm_preds.append( self.relu((preds * hg + feature * hg ) * keys ) )
                if i < self.nstack - 1:
                    x_embed = x_embed + self.merge_preds[i](preds) + self.merge_features[i](feature)
            
            x_combine = torch.cat(combined_hm_preds, dim = 1)
            x_combine =self.final[batch_index](x_combine)
            # print("x_combine:", x_combine.shape)
            batch_item_combined_hm_preds.append(x_combine)
            
        x_combine = torch.stack( batch_item_combined_hm_preds, 0)
        # print("total:x_combin:", x_combine.shape)
        outp = self.re_build( x_combine )
        outp = self.upsample(outp)
        edge = nn.functional.pad(outp, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return self.build_results(outp, edge) if (self.need_return_dict) else (outp, edge)
from .FL_base import FL_base
class FL_FPN(FL_base):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1,
                embed_shape = ( 2, 4),
                batch_size = 16,
                need_return_dict = False
        ):
        super(FL_FPN, self).__init__()
     
        self.batch_size = batch_size
        self.embed_shape = embed_shape
        self.need_return_dict = need_return_dict
        self.middle_channel = middle_channel
       
        # replace your model
        ####################################
        self.model = nn.Sequential(
            FPN()
        )
        ####################################
        self.edge_body = nn.Sequential(
            ResBlock(8,4),
            nn.Conv2d( 4, out_channel, 1, 1),
            nn.ReLU(),
        )
        ####################################
        self.select = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Upsample(scale_factor = 2),
            nn.Conv2d( 32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64,32),
            nn.Conv2d( 32, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        ####################################
        self.consist_stage = nn.Sequential(
            nn.Conv2d( middle_channel, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d( 32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d( 64, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            #nn.Upsample(scale_factor = 2),
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
        x1 = self.consist_stage(x)
        x = x * x1 
        return x
    
    def edge_hot_map(self, x):
        x = x.clone().detach()
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        edge = self.edge_body(edge)
        return edge
    
    def build_results(self, x,y,z):
        return {
            "mask": x,
            "cmask": y,
            "edge":z,
        }
    
    def forward(self, x):   
        x = self.downsample(x)
        # print(x.shape)
        x = self.ext_feature(x)

        outp = self.consist(x)
        # assert False, (x.shape, outp.shape)
        edge = self.edge_hot_map(outp)
        outp = self.select(torch.cat([ outp - edge, edge], 1) )
        outp = self.final(outp)
        outp = self.sigmod(outp)
        # print(outp.shape, edge.shape)
        # outp = self.upsample(outp)
        # assert False, (outp.shape, self.upsample(x).shape, edge.shape)
        return self.build_results(x, outp, edge) if (self.need_return_dict) else ( x, outp, edge)
