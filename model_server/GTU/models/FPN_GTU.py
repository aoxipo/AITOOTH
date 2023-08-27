from .GT_UNet import *
from .GT_UNet import _make_bot_layer
import torch

class Decode(nn.Module):
    def __init__(self, ch_in, ch_out):
        dim_in = ch_in
        dim_out = ch_out
        super(Decode, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d( dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )
        # self.up = nn.Upsample( scale_factor = 2)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class GT_UFPN_Net(nn.Module):
    def __init__(self, img_ch  =  1, output_ch  =  1, 
                middle_channel = [32, 64, 128, 256, 512], 
                encode_len = 3, 
                need_return_dict = False
        ):
        super(GT_UFPN_Net,self).__init__()
        
        self.index_len = encode_len - 1
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2)
        self.Maxpool  =  nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        self.encode_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.decode_list = nn.ModuleList()
        
        middle_channel = middle_channel[ len(middle_channel) - encode_len : ]
        middle_channel =  [ img_ch, *middle_channel]
        print( middle_channel )
        self.pre_encode = _make_bot_layer(ch_in = middle_channel[0], ch_out = middle_channel[ 1 ])
        for i in range(1, encode_len):
            self.encode_list.append( _make_bot_layer(ch_in = 2 * middle_channel[i], ch_out = middle_channel[ i+1 ]) )
        
        for i in range(1, encode_len):
            now_dim = encode_len - i + 1
            next_dim = encode_len - i
            # print( middle_channel[ now_dim ],   middle_channel[ next_dim ])
            self.up_list.append( up_conv(ch_in = middle_channel[now_dim ] , ch_out = 2 * middle_channel[next_dim]) )
            self.decode_list.append( 
                # _make_bot_layer(ch_in = 2 * middle_channel[ now_dim ], ch_out = middle_channel[next_dim]) 
                Decode( ch_in = 2 * middle_channel[ now_dim ], ch_out = middle_channel[next_dim] )
                )
            
        self.CBR = nn.ModuleList()
        for i in range(encode_len):
            self.CBR.append(
                nn.Sequential(
                    nn.Conv2d(middle_channel[0], middle_channel[i+1], 3, 1, 1),
                    nn.BatchNorm2d(middle_channel[i+1]),
                    nn.ReLU(),
                )
            )
        self.last_up = nn.Upsample(scale_factor = 2)
        self.last_decode = nn.Conv2d(middle_channel[next_dim], output_ch, kernel_size = 1, stride = 1, padding = 0)



    def build_results(self, x,y,z):
        return {
            "mask": x,
            "cmask": y,
            "edge":z,
        }
    
    def build_feature_pyramid(self, x): # 80
        x_list = []
        for i in range(self.index_len + 1):
            x = self.downsample(x) 
            x_list.append( x )
        return   x_list
    
    def edge_hot_map(self, x):
        x = x.clone().detach()
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return edge
    

    def forward(self,x):
        x_list = self.build_feature_pyramid( x )
        pre_x_list = []
        out_list = []
        
        for index in range(len(x_list)):
            pre_x = self.CBR[index](x_list[index])
            pre_x_list.append(pre_x)
            # print(x_list[index].shape, pre_x.shape)
        # pre_x_list = [x, *pre_x_list]
        
        out = self.pre_encode( x )
        out_pool  =  self.Maxpool(out)
        out_list.append(out)
        # print( "x{}:{} {}".format( 0, out.shape, out_pool.shape ))
        
        # encoding
        for index in range(self.index_len):
            x_temp = torch.cat( [pre_x_list[index], out_pool], 1) 
            # print( "x{} cat:{}".format( index, x.shape))
            out = self.encode_list[index]( x_temp )
            out_pool = self.Maxpool(out)
            out_list.append(out)
            # print( "x{}:{} {}".format( index, out.shape, out_pool.shape ))
            
        x_temp = out_pool
        # decoding
        for index in range(self.index_len):
            up = self.up_list[index](x_temp)
            x_temp = torch.cat( [up, out_list[ self.index_len - index ] ], dim = 1)
            x_temp = self.decode_list[index](x_temp)
            # print( "decode{}:{} {}".format( index, up.shape, x_temp.shape))
        
        # print("final decode:", x_temp.shape)
        outp = self.last_up(x_temp)
        outp = self.last_decode(outp)
        # print("outp:{}".format( outp.shape ))
        return self.build_results(outp, outp, 0) if self.need_return_dict else( outp, 0 ) 
