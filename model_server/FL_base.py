

import torch
import torch.nn as nn
import torch.nn.functional as F

from .RESUNet import ResBlock
from .util import MinPool, cat_tensor, crop_tensor
from .model import RCS, DCBL, RC
from .model import Unet

class FL_base(nn.Module):
    def __init__(
                self,
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1,
                embed_shape = ( 2, 4),
                batch_size = 16,
                need_return_dict = False
        ):
        super(FL_base, self).__init__()
     
        self.batch_size = batch_size
        self.embed_shape = embed_shape
        self.need_return_dict = need_return_dict
        self.middle_channel = middle_channel
        d = [nn.Identity()] # nn.AvgPool2d(2)]
        if self.middle_channel != 1 :
            d.append(nn.Conv2d(in_channel, middle_channel, 1,1))
        self.downsample = nn.Sequential(
            *d
        )
        self.upsample = nn.Identity() # nn.Upsample(scale_factor = 2)# nn.ConvTranspose2d(out_channel, out_channel, (2,2), (2,2))
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        
        # replace your model
        ####################################
        self.model = nn.Sequential(
            Unet(False)
        )
        ####################################
        
        self.final = nn.Conv2d(middle_channel, out_channel, 1,1 )
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
            # print(batch_item_x_embed.shape)
            
            #### your forward model here
            output, _ = self.model( batch_item_x_embed ) # only for mask not edge, edge will have another way
            #### 
            
            if self.middle_channel != 1:
                output = self.final[batch_index](output)
                
            batch_item_combined_hm_preds.append(output)
            
        x_combine = torch.stack(batch_item_combined_hm_preds, 0)
        # print("total:x_combin:", x_combine.shape)
        outp = self.re_build( x_combine )
        outp = self.upsample(outp)
        edge = nn.functional.pad(outp, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return self.build_results(outp, edge) if (self.need_return_dict) else (outp, edge)
 