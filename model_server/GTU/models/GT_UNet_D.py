
from .pvtv2 import pvt_v2_b2
import torch.nn as nn
import torch
from .GT_UNet import MinPool, ResGroupFormer, up_conv, SKConv, Decode, _make_bot_layer
import math
from einops import rearrange, reduce, repeat

def exists(x):
    return x is not None

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResGroupFormer_ddpm(ResGroupFormer):
    def __init__(self, ch_in, ch_out, time_emb_dim, w_list = [4,4]):
        super().__init__()
        self.g1 = _make_bot_layer(
                    ch_in = ch_in, #  2 * middle_channel[i],             
                    ch_out = ch_out, #middle_channel[ i+1 ], 
                    w = w_list[0]
                ) 
        self.g2 = _make_bot_layer(
                    ch_in = ch_out, # 2 * middle_channel[i],             
                    ch_out = ch_out, #middle_channel[ i+1 ], 
                    w = w_list[1]
                ) 
        self.g3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.GroupNorm(8, ch_out),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, 1, 1),
            nn.GroupNorm(8, ch_out),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, 1, 1),
            nn.GroupNorm(8, ch_out),
            nn.SiLU(),
        )            
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, ch_out)
        ) 
        self.sk = SKConv([ ch_out, ch_out], 
                ch_out, 
                32,2,8,2
            )
        # self.sigmod = nn.Sigmoid()
    def forward(self, x, time_emb = None):

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x_clone = x.clone().detach()
        x = self.g1(x)
        # print( x.shape )
        x = self.g2(x)
        
        y = self.g3(x_clone)
        # x = self.sigmod(x + y)
        x = self.sk((x, y))
        return x

class Decode_ddpm(Decode):
    def __init__(self, ch_in, ch_out, time_emb_dim, decode_type = "conv"):
        dim_in = ch_in
        dim_out = ch_out
        super(Decode_ddpm, self).__init__()
        if decode_type == "conv":
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
            print("detype conv")
        else:
            self.conv = ResGroupFormer( dim_in, dim_out, )
            print("res g former")
        self.up = nn.Upsample( scale_factor = 2)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, ch_out)
        ) 
        
    def forward(self, x, y, time_emb = None):
        y = self.up(y)
        x = torch.cat([x,y], 1)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.conv(x)
        return x


class GT_U_DC_PVTNet(nn.Module):
    def __init__(self, img_ch  =  1, output_ch  =  1, 
                middle_channel = [64, 128, 256, 512, 1024], 
                encode_len = 5, 
                need_return_dict = True,
                need_supervision = False,
                decode_type = "conv",
        ):
        super(GT_U_DC_PVTNet, self).__init__()
        pvt_channel = [64, 128, 320, 512]
        self.index_len = encode_len - 1
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2)
        self.Maxpool  =  nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        self.encode_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.decode_list = nn.ModuleList()
        self.sk_list = nn.ModuleList()
        if self.need_return_dict:
            self.supervision_list = nn.ModuleList()
        # self.select  =  nn.Conv2d( 2, output_ch, kernel_size = 1, stride = 1, padding = 0)
        middle_channel = middle_channel[ len(middle_channel) - encode_len : ]
        middle_channel =  [ img_ch, *middle_channel]
        self.need_supervision = need_supervision
        print( middle_channel )
        self.pre_encode = ResGroupFormer(middle_channel[0], middle_channel[ 1 ]) 
        for i in range(1, encode_len):
            self.encode_list.append( 
                ResGroupFormer( ch_in = 2 * middle_channel[i], ch_out = middle_channel[ i+1 ],  )
            )
        
        for i in range(1, encode_len):
            now_dim = encode_len - i + 1
            next_dim = encode_len - i
            # print( middle_channel[ now_dim ],   middle_channel[ next_dim ])
            self.up_list.append( 
                up_conv(ch_in = middle_channel[now_dim ] , ch_out = 2 * middle_channel[next_dim]) 
            )
            
            if i < self.index_len :
                # print(pvt_channel[ 4 - i - 1 ], middle_channel[now_dim ])
                self.sk_list.append( 
                    SKConv([pvt_channel[ self.index_len - i - 1], middle_channel[now_dim ]], middle_channel[now_dim ], 32,2,8,2)
                )
                
          
            self.decode_list.append( 
                    Decode(ch_in = 2 * middle_channel[ now_dim ], ch_out = middle_channel[next_dim], decode_type =  decode_type) 
                    # SKConv([middle_channel[ now_dim ], middle_channel[now_dim ]], middle_channel[next_dim], 32,2,8,2)
                )
            
            print("sk:",len(self.sk_list))
            if self.need_supervision:
                self.supervision_list.append( nn.Conv2d(middle_channel[next_dim], output_ch, 1, 1) )
            #  _make_bot_layer(
            #     ch_in = 2 * middle_channel[ now_dim ], 
            #     ch_out = middle_channel[next_dim],
            #     w = 2 ** i) 
            #  )
            
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
        if self.need_supervision:
            self.supervision_list.append( nn.Conv2d(middle_channel[next_dim], output_ch, 1, 1) )

        self.select = nn.Sequential(
                nn.Conv2d(2, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d( 32, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d( 16, 1, 1, 1),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            )

    def build_results(self, x,y,z, super_vision  = None):
        if super_vision is None:
            return {
            "mask": x,
            "cmask": y,
            "edge":z,
        }
        else:
            return {
                "mask": x,
                "cmask": y,
                "edge":z,
                "super":super_vision
            }
            
    
    def build_feature_pyramid(self, x): # 80
        x_list = []
        for i in range(self.index_len + 1):
            x = self.downsample(x) 
            x_list.append( x )
        return   x_list
    
    def edge_hot_map(self, x1):
        x = x1.clone().detach()
        # x[x < 0.5] = 0
        # x[x >= 0.5] = 1
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        # edge = self.edge_hot(edge)
        return edge
    
    def forward(self,x):

        x_list = self.build_feature_pyramid( x )
        pre_x_list = []
        out_list = []
        supervision = []
        for index in range(len(x_list)):
            pre_x = self.CBR[index](x_list[index])
            pre_x_list.append(pre_x)
            # print(x_list[index].shape, pre_x.shape)
        # pre_x_list = [x, *pre_x_list]
       
        out = self.pre_encode( x )
        out_pool = self.Maxpool(out)
        out_list.append(out)
        # print( "x{}:{} {}".format( 0, out.shape, out_pool.shape ))
        
        # encoding
        for index in range(self.index_len):
            x_temp = torch.cat( [pre_x_list[index], out_pool], 1) 
            # print( "x{} cat:{}".format( index, x.shape))
            out = self.encode_list[index]( x_temp )
            out_pool = self.Maxpool(out)
            out_list.append(out)
            
            # print( "x{}:{} {}".format( index + 1, out.shape, out_pool.shape ))
            
        x_temp = out_pool 
        # decoding
        for index in range(len(self.sk_list)):
            up = out_list[ self.index_len - index ]
            # print(up.shape, pvt_decode[index].shape)
            conmbine_feature = self.sk_list[index](( pvt_decode[index] , up))
            out_list[ self.index_len - index] = conmbine_feature
            # print("index:", index, self.index_len - index, "pvt and up", pvt_decode[index].shape, up.shape)
        
            
        for index in range(self.index_len):
            # up = self.up_list[index](x_temp)
            # x_temp = torch.cat( [up, out_list[ self.index_len - index ] ], dim = 1)
            up = out_list[ self.index_len - index ]
           
            # print("decode (up, x_temp):", up.shape, x_temp.shape)
            x_temp = self.decode_list[index](up, x_temp)
            
            
            # x_temp = self.decode_list[index](x_temp)
            supervision.append(x_temp)
            # self.need_supervision[index](x_temp)
            # print( "decode{}:{} {}".format( index, up.shape, x_temp.shape))
        
        # print("final decode:", x_temp.shape)
        outp = self.last_up(x_temp)
        out = self.last_decode(outp)
        
        if self.need_return_dict == False:
            return out, outp
        
        edge = self.edge_hot_map(out)
        outp = self.select(torch.cat([out, edge], 1))



        # print("outp:{}".format( outp.shape ))
        if self.need_supervision:
            for i in range( self.index_len ):
                supervision[i] = self.supervision_list[i](supervision[i])
            return self.build_results(outp, outp, edge, supervision) if self.need_return_dict else( outp, edge, supervision) 
        return self.build_results(outp, out, edge) if self.need_return_dict else( outp, edge ) 

class GT_U_DC_ddpm(nn.Module):
    def __init__(self, img_ch  =  1, output_ch  =  1, 
                middle_channel = [64, 128, 256, 512, 1024], 
                encode_len = 5, 
                need_return_dict = True,
                need_supervision = False,
                decode_type = "conv",
        ):
        super(GT_U_DC_PVTNet, self).__init__()
        pvt_channel = [64, 128, 320, 512]
        self.index_len = encode_len - 1
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2)
        self.Maxpool  =  nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        self.encode_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.decode_list = nn.ModuleList()

        if self.need_return_dict:
            self.supervision_list = nn.ModuleList()

        middle_channel = middle_channel[ len(middle_channel) - encode_len : ]
        middle_channel =  [ img_ch, *middle_channel]
        self.need_supervision = need_supervision
        time_dim = 64 * 4
        print( middle_channel )
        self.pre_encode = ResGroupFormer_ddpm(
            middle_channel[0], middle_channel[ 1 ],
            time_emb_dim = time_dim
            ) 
        for i in range(1, encode_len):
            self.encode_list.append( 
                ResGroupFormer_ddpm( 
                ch_in = 2 * middle_channel[i], 
                ch_out = middle_channel[ i+1 ],  

                time_emb_dim = time_dim)
            )
        
        for i in range(1, encode_len):
            now_dim = encode_len - i + 1
            next_dim = encode_len - i
            # print( middle_channel[ now_dim ],   middle_channel[ next_dim ])
            self.up_list.append( 
                up_conv(ch_in = middle_channel[now_dim ] , ch_out = 2 * middle_channel[next_dim]) 
            )
            self.decode_list.append( 
                    Decode_ddpm(
                ch_in = 2 * middle_channel[ now_dim ],
                  ch_out = middle_channel[next_dim], 
                  time_emb_dim = time_dim,
                  decode_type =  decode_type,
                  ) 
                    # SKConv([middle_channel[ now_dim ], middle_channel[now_dim ]], middle_channel[next_dim], 32,2,8,2)
                )
            
            print("sk:",len(self.sk_list))
            if self.need_supervision:
                self.supervision_list.append( nn.Conv2d(middle_channel[next_dim], output_ch, 1, 1) )
            #  _make_bot_layer(
            #     ch_in = 2 * middle_channel[ now_dim ], 
            #     ch_out = middle_channel[next_dim],
            #     w = 2 ** i) 
            #  )
            
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
        if self.need_supervision:
            self.supervision_list.append( nn.Conv2d(middle_channel[next_dim], output_ch, 1, 1) )

        self.select = nn.Sequential(
                nn.Conv2d(2, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d( 32, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d( 16, 1, 1, 1),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            )


        sinu_pos_emb = SinusoidalPosEmb(time_dim//4)
        fourier_dim = time_dim//4
        self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
    def build_results(self, x,y,z, super_vision  = None):
        if super_vision is None:
            return {
            "mask": x,
            "cmask": y,
            "edge":z,
        }
        else:
            return {
                "mask": x,
                "cmask": y,
                "edge":z,
                "super":super_vision
            }
            
    
    def build_feature_pyramid(self, x): # 80
        x_list = []
        for i in range(self.index_len + 1):
            x = self.downsample(x) 
            x_list.append( x )
        return   x_list
    
    def edge_hot_map(self, x1):
        x = x1.clone().detach()
        # x[x < 0.5] = 0
        # x[x >= 0.5] = 1
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        # edge = self.edge_hot(edge)
        return edge
    
    def forward(self,x, time):
        t = self.time_mlp(time)

        x_list = self.build_feature_pyramid( x )
        pre_x_list = []
        out_list = []
        supervision = []
        for index in range(len(x_list)):
            pre_x = self.CBR[index](x_list[index])
            pre_x_list.append(pre_x)
            # print(x_list[index].shape, pre_x.shape)
        # pre_x_list = [x, *pre_x_list]
       
        out = self.pre_encode( x )
        out_pool = self.Maxpool(out)
        out_list.append(out)
        # print( "x{}:{} {}".format( 0, out.shape, out_pool.shape ))
        
        # encoding
        for index in range(self.index_len):
            x_temp = torch.cat( [pre_x_list[index], out_pool], 1) 
            # print( "x{} cat:{}".format( index, x.shape))
            out = self.encode_list[index]( x_temp , t)
            out_pool = self.Maxpool(out)
            out_list.append(out)
            
            # print( "x{}:{} {}".format( index + 1, out.shape, out_pool.shape ))
            
        x_temp = out_pool 
        # decoding
        # for index in range(len(self.sk_list)):
        #     up = out_list[ self.index_len - index ]
        #     # print(up.shape, pvt_decode[index].shape)
        #     conmbine_feature = self.sk_list[index](( pvt_decode[index] , up))
        #     out_list[ self.index_len - index] = conmbine_feature
            # print("index:", index, self.index_len - index, "pvt and up", pvt_decode[index].shape, up.shape)
        
            
        for index in range(self.index_len):
            # up = self.up_list[index](x_temp)
            # x_temp = torch.cat( [up, out_list[ self.index_len - index ] ], dim = 1)
            up = out_list[ self.index_len - index ]
           
            # print("decode (up, x_temp):", up.shape, x_temp.shape)
            x_temp = self.decode_list[index](up, x_temp, t)
            
            
            # x_temp = self.decode_list[index](x_temp)
            supervision.append(x_temp)
            # self.need_supervision[index](x_temp)
            # print( "decode{}:{} {}".format( index, up.shape, x_temp.shape))
        
        # print("final decode:", x_temp.shape)
        outp = self.last_up(x_temp)
        out = self.last_decode(outp)
        
        if self.need_return_dict == False:
            return out, outp
        
        edge = self.edge_hot_map(out)
        outp = self.select(torch.cat([out, edge], 1))



        # print("outp:{}".format( outp.shape ))
        if self.need_supervision:
            for i in range( self.index_len ):
                supervision[i] = self.supervision_list[i](supervision[i])
            return self.build_results(outp, outp, edge, supervision) if self.need_return_dict else( outp, edge, supervision) 
        return self.build_results(outp, out, edge) if self.need_return_dict else( outp, edge ) 