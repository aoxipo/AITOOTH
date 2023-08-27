import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import MinPool
from .model import *
from .util import cat_tensor, crop_tensor
from torchsummary import summary
from .util import MinPool, cat_tensor, crop_tensor
from .FL_base import FL_base
from .GTU.models.FPN_GTU import GT_UFPN_Net
from .GTU.models.GT_UNet import GT_U_DC_PVTNet
from .FL_DETR import FPN
class FL_GTU(FL_base):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                encode_len = 3, 
                need_return_dict = False
        ):
        super(FL_GTU, self).__init__()
        self.index_len = encode_len - 1
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2)
        self.model_1 = GT_U_DC_PVTNet(img_ch  =  1, middle_channel = [64, 128, 256, 512, 1024], encode_len = 4, 
                                      need_return_dict = False)  
        # self.model_2 = GT_UFPN_Net(img_ch  =  3, middle_channel = [32, 64, 128, 256])  
        self.model_3 = GT_UFPN_Net(img_ch  =  1, middle_channel = [64, 128, 256, 512], need_return_dict = False)  
        # self.model_3 = GT_UFPN_Net(middle_channel = [64, 128, 256, 512, 1024], encode_len = 4)  
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
        
        feature = self.ext_feature_batch(x[0], 4, 8)
        attn_map = self.re_build_detail(feature, 4, 8)
        
        # feature = self.ext_feature_batch(x[0], 2, 4, attention_map = attn_map)
        # attn_map = self.re_build_detail(feature, 2, 4)

        feature = self.ext_feature_batch(x[0], 1, 2, attention_map = attn_map)
        hot_map = self.re_build_detail(feature, 1, 2)
        
        return hot_map

    def ext_feature_batch(self, x, w, h, attention_map = None):
        
        if attention_map is not None:
            x = x * attention_map
        
        # print(x.shape, w, h)
        x_embed = self.get_embeding_detail(x, w, h)
        BB, B, C, W, H =  x_embed.shape
        
        batch_item_combined_hm_preds = []
        for batch_index in range(BB): 

            batch_item_x_embed = x_embed[batch_index,:,:,:,:]
            
            #### your forward model here
            if w ==  1:
                output, _ = self.model_1( batch_item_x_embed )
            # if w ==  2:
            #     output, _ = self.model_2( batch_item_x_embed )
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
        edge = self.dilate(edge) - self.erode(edge)
        edge = self.edge_body(edge)
        return edge
    
    def forward(self, x):  
        x_list = self.build_feature_pyramid(x)
        out = self.ext_feature(x_list)
        outp = self.consist(out)
        # assert False, (x.shape, outp.shape)
        edge = self.edge_hot_map(outp)
        outp = self.select(torch.cat([ outp - edge, edge], 1) )
        # op = torch.cat([ outp - outp * edge, outp * edge], 1)
        # print(op.shape, outp.shape, edge.shape)
        # outp = self.select( op )
        outp = self.final(outp)
        outp = self.sigmod(outp)
        return self.build_results(out, outp, edge) if self.need_return_dict else (out, outp, edge)
    
    def forward_consist(self, x, iter = 2):
        x_list = self.build_feature_pyramid(x)
        
        out = self.ext_feature(x_list)
        outp = self.consist(out)
        # assert False, (x.shape, outp.shape)
        edge = self.edge_hot_map(outp)
        outp = self.select(torch.cat([ outp - edge, edge], 1) )
        # op = torch.cat([ outp - outp * edge, outp * edge], 1)
        # print(op.shape, outp.shape, edge.shape)
        # outp = self.select( op )
        outp = self.final(outp)
        outp = self.sigmod(outp)

        attn_map = outp
        
        for i in range(iter):
            
            attn_map[attn_map < 0.5 ] = 0
            attn_map[attn_map > 0.5 ] = 1

            feature = self.ext_feature_batch(x_list[0], 1, 2, attention_map = attn_map)
            attn_map = self.re_build_detail(feature, 1, 2)

        outp = attn_map
        return self.build_results(out, outp, edge) if self.need_return_dict else (out, outp, edge)


