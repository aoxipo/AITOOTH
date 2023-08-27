import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import MinPool
from .RESUNet import ResBlock
from .model import *
from .util import cat_tensor, crop_tensor
from .FL_seris import Encode, Decode
from torchsummary import summary
from .util import MinPool, cat_tensor, crop_tensor
from .FL_base import FL_base
class FPN(nn.Module):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                # block_layers=[6, 12, 24, 16], 
                # transition_layer = [256, 512, 1024, 1024],
                middle_channel = [8, 16, 32, 64, 128],
                encode_len = 4,
                need_return_dict = False
        ):
        super(FPN,self).__init__()
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2,2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        middle_channel = middle_channel[ len(middle_channel) - encode_len : ]
        # print("middle_channel:", middle_channel)
        index_len = encode_len - 1
    
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
            )  for i in range(index_len) ]
        )
        self.decode = nn.ModuleList(
            [
                Decode(
                    2 * middle_channel[index_len - i], 
                    2 * middle_channel[index_len - i - 1], 
                    conv_type = "conv") 
                for i in range(index_len)
            ]
        )
        self.CBR = nn.ModuleList()
        for i in range(encode_len):
            self.CBR.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, middle_channel[i],3, 1, 1),
                    nn.BatchNorm2d(middle_channel[i]),
                    nn.ReLU(),
                )
            )
        self.index_len = index_len     
        
    def build_feature_pyramid(self, x): # 80
        x_list = []
        x_list.append(x)
        for i in range(self.index_len + 1):
            x = self.downsample(x) 
            x_list.append( x )
        return   x_list

    def feature(self, x):
        x_encode_list = []
        for i in range(self.index_len + 1):
            x_encode_list.append( self.CBR[i]( x[ i + 1 ] ) )
            # print(x_encode_list[-1].shape)
        xc_list = []
        xp_list = []
        
        xc_0, xp_0 = self.pre_encode(x[0])
        xc_list.append(xc_0)
        xp_list.append(xp_0)
        
        for i in range(self.index_len):
            x_cat  = torch.cat([xp_list[i], x_encode_list[i]], 1)
            ec, ep = self.encode[i](x_cat)
            # print(ec.shape, ep.shape, )
            xc_list.append(ec)
            xp_list.append(ep)
        
        x_c = torch.cat([xp_list[-1], x_encode_list[-1]], 1)
        # print("cat :", x_c.shape)
        xc_list = list(reversed( xc_list ))
        
        for i in range(self.index_len):
            x_c = self.decode[i](x_c, xc_list[i])
            # print("decode :", x_c.shape)
        
        x_c = self.last_decode(x_c, xc_0)
        # print("decode :", x_c.shape)
        out = self.out(x_c)
        edge = nn.functional.pad(out, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return  out, edge
    
    def forward(self, x):
        x = self.feature(self.build_feature_pyramid(x))
        return x

class Unet(nn.Module):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                # block_layers=[6, 12, 24, 16], 
                # transition_layer = [256, 512, 1024, 1024],
                middle_channel = [16, 32, 64, 128],
              
        ):
        super(Unet,self).__init__()
   
        self.pre_encode = nn.Sequential(
            Encode(in_channel, middle_channel[0], 4)
        )
        self.out = nn.Sequential(
            nn.Conv2d( middle_channel[0], out_channel,1,1)
        )
        self.brige = nn.Sequential(
            nn.Conv2d(middle_channel[-1], 2 * middle_channel[-1], 1,1),
            nn.BatchNorm2d(2 * middle_channel[-1]),
            nn.ReLU()
        )
        self.brige1 = nn.Sequential(
            nn.Conv2d(middle_channel[-1], middle_channel[-1], 1,1),
            nn.BatchNorm2d(middle_channel[-1]),
            nn.ReLU()
        )
        self.last_decode = Decode(middle_channel[1], middle_channel[0], conv_type = "conv") 
        self.encode = nn.ModuleList(
            [ Encode(
                middle_channel[i], 
                middle_channel[ i+1 ], 2, 
                conv_type = "conv"
            )  for i in range(3) ]
        )
        self.decode = nn.ModuleList(
            [
                Decode(
                    2 * middle_channel[3 - i], 
                    2 * middle_channel[3 - i - 1], 
                    conv_type = "conv") 
                for i in range(3)
            ]
        )
        
    def feature(self, x):
        xc_0, xp_0 = self.pre_encode(x)
        ec_0, ep_0 = self.encode[0](xp_0)
        ec_1, ep_1 = self.encode[1](ep_0)
        ec_2, ep_2 = self.encode[2](ep_1)
        
        x_m = self.brige(ep_2)
        x_n = self.brige1(ec_2)

        d_0 = self.decode[0](x_m, x_n)
        d_1 = self.decode[1](d_0, ec_1)
        d_2 = self.decode[2](d_1, ec_0)
        d_3 = self.last_decode(d_2, xc_0)
        out = self.out(d_3)
        return  out, 0
    
    
    def forward(self, x):
        x= self.feature(x)
        return x
    
class FL_DETR(FL_base):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                encode_len = 4, 
                need_return_dict = False
        ):
        super(FL_DETR, self).__init__()
        self.index_len = encode_len - 1
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2)
        self.model_1 = FPN() # Unet()
        self.model_2 = FPN() # Unet()
        self.model_3 = FPN(middle_channel = [16, 32, 64, 128, 256]) #Unet()
        self.final = nn.Conv2d(8, out_channel, 1,1 )
        self.edge_body = nn.Sequential(
            ResBlock(8,4),
            nn.Conv2d( 4, out_channel, 1, 1),
            nn.ReLU(),
        )
        self.consist_stage = nn.Sequential(
            nn.Conv2d( 1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d( 32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d( 16, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            #nn.Upsample(scale_factor = 2),
        )
        self.select = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Upsample(scale_factor = 2),
            nn.Conv2d( 32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d( 16, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def build_feature_pyramid(self, x): # 80
        x_list = []
        x_list.append(x)
        for i in range(self.index_len):
            x = self.downsample(x) 
            x_list.append( x )
        #         for i in range(self.index_len):
        #             print(x_list[i].shape)
        return   x_list
    
    def get_embeding_detail(self, x, w, h):
        x_re1 = crop_tensor(x, w, h)   
        return x_re1
    
    def re_build_detail(self, x, w, h):
        x_re1 = cat_tensor(x, w, h)   
        return x_re1
    
    def ext_feature(self, x):
        # x_feature_list = []
        
        feature = self.ext_feature_batch(x[2], 1, 2)
        attn_map = self.re_build_detail(feature, 1, 2)
        
        feature = self.ext_feature_batch(x[1], 2, 4, attention_map = attn_map)
        attn_map = self.re_build_detail(feature, 2, 4)
        
        feature = self.ext_feature_batch(x[0], 4, 8, attention_map = attn_map)
        hot_map = self.re_build_detail(feature, 4, 8)
        
        return hot_map

    def ext_feature_batch(self, x, w, h, attention_map = None):
        
        if attention_map is not None:
            x = x * self.upsample(attention_map)
        
        # print(x.shape, w, h)
        x_embed = self.get_embeding_detail(x, w, h)
        BB, B, C, W, H =  x_embed.shape
        
        batch_item_combined_hm_preds = []
        for batch_index in range(BB): 

            batch_item_x_embed = x_embed[batch_index,:,:,:,:]
            #### your forward model here
            if w ==  1:
                output, _ = self.model_1( batch_item_x_embed )
            if w ==  2:
                output, _ = self.model_2( batch_item_x_embed )
            if w ==  4:
                output, _ = self.model_3( batch_item_x_embed )
            #### 
            batch_item_combined_hm_preds.append(output)

        x_combine = torch.stack(batch_item_combined_hm_preds, 0)
        return x_combine
    

    def build_results(self, x,y,z):
        return {
            "mask": x,
            "cmask": y,
            "edge":z,
        }
    
    def consist(self, x):
        x = x.clone().detach()
        x1 = self.consist_stage(x)
        return x1

    def edge_hot_map(self, x):
        x = x.clone().detach()
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge_old = self.dilate(edge) - self.erode(edge)
        edge = self.edge_body(edge_old)
        return edge, edge_old
    
    def forward(self, x):  
        x_list = self.build_feature_pyramid(x)
        out = self.ext_feature(x_list)
        outp = self.consist(x)
        # assert False, (x.shape, outp.shape)
        edge, edge_old = self.edge_hot_map(outp)
        outp = self.select(torch.cat([ outp - edge_old, edge], 1) )
        # op = torch.cat([ outp - outp * edge, outp * edge], 1)
        # print(op.shape, outp.shape, edge.shape)
        # outp = self.select( op )
        outp = self.final(outp)
        outp = self.sigmod(outp)
        return self.build_results(out, outp, edge) if self.need_return_dict else (out, outp, edge)

