import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


BATCH_NORM_DECAY  =  1 - 0.9  # pytorch batch norm `momentum  =  1 - counterpart` of tensorflow
BATCH_NORM_EPSILON  =  1e-5


def get_act(activation):
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation  ==  'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish


class BNReLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation = 'relu', nonlinearity = True, init_zero = False):
        super(BNReLU, self).__init__()

        self.norm  =  nn.BatchNorm2d(out_channels, momentum = BATCH_NORM_DECAY, eps = BATCH_NORM_EPSILON)
        if nonlinearity:
            self.act  =  get_act(activation)
        else:
            self.act  =  None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out  =  self.norm(input)
        if self.act is not None:
            out  =  self.act(out)
        return out
    
class GNSiLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation = 'relu', nonlinearity = True, init_zero = False):
        super(GNSiLU, self).__init__()
        self.norm  = nn.GroupNorm(8, out_channels)
        if nonlinearity:
            self.act  =  nn.SiLU()
        else:
            self.act  =  None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out  =  self.norm(input)
        if self.act is not None:
            out  =  self.act(out)
        return out


class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative = True, fold_heads = False):
        super(RelPosSelfAttention, self).__init__()
        self.relative  =  relative
        self.fold_heads  =  fold_heads
        self.rel_emb_w  =  nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h  =  nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std = dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std = dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim  =  q.shape
        q  =  q * (dim ** -0.5)  # scaled dot-product
        logits  =  torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits +=  self.relative_logits(q)
        weights  =  torch.reshape(logits, [-1, heads, h, w, h * w])
        weights  =  F.softmax(weights, dim = -1)
        weights  =  torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out  =  torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out  =  torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w  =  self.relative_logits_1d(q, self.rel_emb_w, transpose_mask = [0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h  =  self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask = [0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim  =  q.shape
        rel_logits  =  torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits  =  torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits  =  self.rel_to_abs(rel_logits)
        rel_logits  =  torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits  =  torch.unsqueeze(rel_logits, dim = 3)
        rel_logits  =  rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits  =  rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _  =  x.shape
        col_pad  =  torch.zeros((bs, heads, length, 1), dtype = x.dtype).to(x.device)
        x  =  torch.cat([x, col_pad], dim = 3)
        flat_x  =  torch.reshape(x, [bs, heads, -1]).to(x.device)
        flat_pad  =  torch.zeros((bs, heads, length - 1), dtype = x.dtype).to(x.device)
        flat_x_padded  =  torch.cat([flat_x, flat_pad], dim = 2)
        final_x  =  torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x  =  final_x[:, :, :length, length - 1:]
        return final_x


class AbsPosSelfAttention(nn.Module):

    def __init__(self, W, H, dkh, absolute = True, fold_heads = False):
        super(AbsPosSelfAttention, self).__init__()
        self.absolute  =  absolute
        self.fold_heads  =  fold_heads

        self.emb_w  =  nn.Parameter(torch.Tensor(W, dkh))
        self.emb_h  =  nn.Parameter(torch.Tensor(H, dkh))
        nn.init.normal_(self.emb_w, dkh ** -0.5)
        nn.init.normal_(self.emb_h, dkh ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim  =  q.shape
        q  =  q * (dim ** -0.5)  # scaled dot-product
        logits  =  torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        abs_logits  =  self.absolute_logits(q)
        if self.absolute:
            logits +=  abs_logits
        weights  =  torch.reshape(logits, [-1, heads, h, w, h * w])
        weights  =  F.softmax(weights, dim = -1)
        weights  =  torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out  =  torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out  =  torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def absolute_logits(self, q):
        """Compute absolute position enc logits."""
        emb_h  =  self.emb_h[:, None, :]
        emb_w  =  self.emb_w[None, :, :]
        emb  =  emb_h + emb_w
        abs_logits  =  torch.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits


class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads = 4, proj_factor = 1, target_dimension = None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels  =  target_dimension // proj_factor
        else:
            proj_channels  =  in_channels // proj_factor
        self.w  =  nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std = 0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input  =  input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out  =  torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):


    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type = 'relative', use_pos = True):
        super(MHSA, self).__init__()
        self.q_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)
        self.k_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)
        self.v_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type  ==  'relative':
            self.self_attention  =  RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads = True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q  =  self.q_proj(input)
        k  =  self.k_proj(input)
        v  =  self.v_proj(input)

        o  =  self.self_attention(q = q, k = k, v = v)
        return o


class BotBlock(nn.Module):

    def __init__(self, in_dimension, curr_h, curr_w, proj_factor = 4, activation = 'relu', pos_enc_type = 'relative',
                 stride = 1, target_dimension = None, msha_iter = 1):
        super(BotBlock, self).__init__()
        self.w = curr_w
        if stride !=  1 or in_dimension !=  target_dimension:
            self.shortcut  =  nn.Sequential(
                nn.Conv2d(in_dimension, target_dimension, kernel_size = 3, padding = 1, stride = stride),
                GNSiLU(target_dimension, activation = activation, nonlinearity = True),
            )
        else:
            self.shortcut  =  None

        bottleneck_dimension  =  target_dimension // proj_factor
        self.conv1  =  nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size = 3, padding = 1, stride = 1 ),
            GNSiLU(bottleneck_dimension, activation = activation, nonlinearity = True)
        )

        self.mhsa  =  nn.ModuleList()
        for i in range(msha_iter):
            self.mhsa.append( 
                    MHSA(
                    in_channels = bottleneck_dimension, 
                    heads = 4, curr_h = curr_h, curr_w = curr_w,
                    pos_enc_type = pos_enc_type
                )
            )
        conv2_list  =  []
        if stride !=  1:
            assert stride  ==  2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(GNSiLU(bottleneck_dimension, activation = activation, nonlinearity = True))
        
        self.conv2  =  nn.Sequential(*conv2_list)

        self.conv3  =  nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size = 3,padding = 1, stride = 1),
            GNSiLU(target_dimension, nonlinearity = False, init_zero = True),
        )
        self.last_act  =  get_act(activation)
        self.sigmod = nn.Sigmoid()


    def forward(self, x):
        if self.shortcut is not None:
            shortcut  =  self.shortcut(x)
        else:
            shortcut  =  x
        Q_h  =  Q_w  =  self.w
        N, C, H, W  =  x.shape
        P_h, P_w  =  H // Q_h, W // Q_w
        # print( x.shape)
        x  =  x.reshape(N * P_h * P_w, C, Q_h, Q_w)

        out = self.conv1(x)
        out = self.mhsa[0](out)
        # out_o = out.clone().detach()   
        # out  =  self.mhsa[0](out)
        # out_1  =  self.mhsa[1](out_o)
        # out = self.sigmod (out_1 + out)

        out  =  out.permute(0, 3, 1, 2)  # back to pytorch dim order

        out  =  self.conv2(out)
        out  =  self.conv3(out)

        N1, C1, H1, W1  =  out.shape
        out  =  out.reshape(N, C1, int(H), int(W))

        out +=  shortcut
        out  =  self.last_act(out)

        return out

def init_weights(net, init_type = 'normal', gain = 0.02):
    def init_func(m):
        classname  =  m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') !=  -1 or classname.find('Linear') !=  -1):
            if init_type  ==  'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type  ==  'xavier':
                init.xavier_normal_(m.weight.data, gain = gain)
            elif init_type  ==  'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type  ==  'orthogonal':
                init.orthogonal_(m.weight.data, gain = gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') !=  -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv  =  nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
    def forward(self,x):
        x  =  self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up  =  nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(ch_in,ch_out,kernel_size = 3,stride = 1,padding = 1,bias = True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU()
        )

    def forward(self,x):
        x  =  self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t = 2):
        super(Recurrent_block,self).__init__()
        self.t  =  t
        self.ch_out  =  ch_out
        self.conv  =  nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size = 3,stride = 1,padding = 1,bias = True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU()
        )

    def forward(self,x):
        for i in range(self.t):

            if i == 0:
                x1  =  self.conv(x)
            
            x1  =  self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t = 2):
        super(RRCNN_block,self).__init__()
        self.RCNN  =  nn.Sequential(
            Recurrent_block(ch_out,t = t),
            Recurrent_block(ch_out,t = t)
        )
        self.Conv_1x1  =  nn.Conv2d(ch_in,ch_out,kernel_size = 1,stride = 1,padding = 0)

    def forward(self,x):
        x  =  self.Conv_1x1(x)
        x1  =  self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv  =  nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self,x):
        x  =  self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g  =  nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1,stride = 1,padding = 0,bias = True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x  =  nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1,stride = 1,padding = 0,bias = True),
            nn.BatchNorm2d(F_int)
        )

        self.psi  =  nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1,stride = 1,padding = 0,bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu  =  nn.ReLU()
        
    def forward(self,g,x):
        g1  =  self.W_g(g)
        x1  =  self.W_x(x)
        psi  =  self.relu(g1+x1)
        psi  =  self.psi(psi)
        return x*psi


def _make_bot_layer(ch_in, ch_out, w = 4):

    W  =  H  =  w
    dim_in  =  ch_in
    dim_out  =  ch_out

    stage5  =  []

    stage5.append(
        BotBlock(in_dimension = dim_in, curr_h = H, curr_w = W, stride = 1 , target_dimension = dim_out)
    )

    return nn.Sequential(*stage5)


class MinPool(nn.Module):
    def __init__(self, kernel_size, ndim=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool, self).__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                                  return_indices=return_indices, ceil_mode=ceil_mode)
    def forward(self, x):
        x = self.pool(-x)
        return -x

class GT_U_Net(nn.Module):
    def __init__(self, img_ch  =  3, output_ch  =  1, need_return_dict = True, need_supervision = False):
        super(GT_U_Net,self).__init__()
        self.need_return_dict = need_return_dict
        self.Maxpool  =  nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        self.Conv1  =  _make_bot_layer(ch_in = img_ch, ch_out = 64)
        self.Conv2  =  _make_bot_layer(ch_in = 64, ch_out = 128)
        self.Conv3  =  _make_bot_layer(ch_in = 128, ch_out = 256)
        self.Conv4  =  _make_bot_layer(ch_in = 256, ch_out = 512)
        self.Conv5  =  _make_bot_layer(ch_in = 512, ch_out = 1024)

        self.Up5  =  up_conv(ch_in = 1024, ch_out = 512)
        self.Up_conv5  =  _make_bot_layer(ch_in = 1024, ch_out = 512)

        self.Up4  =  up_conv(ch_in = 512, ch_out = 256)
        self.Up_conv4  =  _make_bot_layer(ch_in = 512, ch_out = 256)
        
        self.Up3  =  up_conv(ch_in = 256, ch_out = 128)
        self.Up_conv3  =  _make_bot_layer(ch_in = 256, ch_out = 128)
        
        self.Up2  =  up_conv(ch_in = 128, ch_out = 64)
        self.Up_conv2  =  _make_bot_layer(ch_in = 128, ch_out = 64)

        self.Conv_1x1  =  nn.Conv2d( 64, output_ch, kernel_size = 1, stride = 1, padding = 0)
        self.select  =  nn.Conv2d( 2, output_ch, kernel_size = 1, stride = 1, padding = 0)
        self.need_supervision = need_supervision
        if need_supervision:
            self.conv  =  nn.Sequential(
                nn.Conv2d(256, 256, kernel_size = 3,stride = 1,padding = 1,bias = True),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )


    def build_results(self, x,y,z):
        return {
            "mask": x,
            "cmask": y,
            "edge":z,
        }
    
    def edge_hot_map(self, x):
        # x = x.clone().detach()
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return edge
    

    def forward(self,x):
        # encoding path
        x1  =  self.Conv1(x)

        x2  =  self.Maxpool(x1)
        x2  =  self.Conv2(x2)
        
        x3  =  self.Maxpool(x2)
        x3  =  self.Conv3(x3)

        x4  =  self.Maxpool(x3)
        x4  =  self.Conv4(x4)

        x5  =  self.Maxpool(x4)
        x5  =  self.Conv5(x5)

        # decoding + concat path
        d5  =  self.Up5(x5)
        d5  =  torch.cat((x4,d5),dim = 1)
        
        d5  =  self.Up_conv5(d5)
        
        d4  =  self.Up4(d5)
        d4  =  torch.cat((x3,d4),dim = 1)
        d4  =  self.Up_conv4(d4)

        d3  =  self.Up3(d4)
        d3  =  torch.cat((x2,d3),dim = 1)
        d3  =  self.Up_conv3(d3)

        d2  =  self.Up2(d3)
        d2  =  torch.cat((x1,d2),dim = 1)
        d2  =  self.Up_conv2(d2)

        d1  =  self.Conv_1x1(d2)

        edge = self.edge_hot_map(d1)
        out = self.select(torch.cat([d1 - edge, edge], 1))

        return self.build_results(d1, out, edge) if self.need_return_dict else( d1, out, edge ) 
    
class Decode(nn.Module):
    def __init__(self, ch_in, ch_out, decode_type = "conv"):
        dim_in = ch_in
        dim_out = ch_out
        super(Decode, self).__init__()
        if decode_type == "conv":
            self.conv = nn.Sequential(
                nn.Conv2d( dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1),
                GNSiLU(dim_out),
                nn.Conv2d(dim_out, dim_out, kernel_size = 1, stride = 1, padding = 0),
                GNSiLU(dim_out),
                nn.Conv2d(dim_out, dim_out, kernel_size = 1, stride = 1, padding = 0),
                GNSiLU(dim_out),
            )
            print("detype conv")
        else:
            self.conv = ResGroupFormer( dim_in, dim_out, )
            print("res g former")
        self.up = nn.Upsample( scale_factor = 2)
        
    def forward(self, x, y):
        y = self.up(y)
        x = torch.cat([x,y], 1)
        x = self.conv(x)
        return x

        

class ResGroupFormer(nn.Module):
    def __init__(self, ch_in, ch_out, w_list = [4,4,4]):
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
        # self.g3 = _make_bot_layer(
        #             ch_in = ch_in, #, 2 * middle_channel[i],             
        #             ch_out = ch_out,# , middle_channel[ i+1 ], 
        #             w = w_list[2]
        #         )
               

        self.sk = SKConv([ ch_out, ch_out], 
                ch_out, 
                32,2,8,2
            )    
        # self.sigmod = nn.Sigmoid()
    def forward(self, x):
        x_clone = x.clone().detach()
        x = self.g1(x)
        x = self.g2(x)
        y = self.g3(x_clone)
        x = self.sk((x,y))
        return x

class SKConv(nn.Module):
    #                  64       32   2  8  2
    def __init__(self, features_list, out_features, WH, M, G, r, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32
        """
        super(SKConv, self).__init__()
        features = features_list[-1]
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features_list[i], out_features, kernel_size=3, stride=1, padding=1, groups=G),
                nn.BatchNorm2d(out_features),
                # nn.ReLU()
            ))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList()
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x[i]).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
            # print(i)
        # print(feas.shape)
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        # print(fea_U.shape, fea_s.shape)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
            # print(i)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        # if attention_vectors.shape[0] != 1 :
        #     attention_vectors = attention_vectors.unsqueeze(0)
        # print( attention_vectors.shape , feas.shape )
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class DSK(nn.Module):
    def __init__(self, in_dim, out_dim, WH = 32):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample( scale_factor = 2),
            nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.skconv = nn.Sequential(
            SKConv( out_dim, WH, 2, 8, 2),
            # nn.Conv2d(out_dim, out_dim, kernel_size = 1, stride = 1, padding = 0),
        )
        self.after_skconv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            nn.Conv2d(out_dim, out_dim, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            nn.Conv2d(out_dim, out_dim, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
    def forward(self, x, y):
        y_up = self.up(y)
        x_conv = self.conv(x)
        # print(y_up.shape, x_conv.shape)
        out = self.skconv([x_conv, y_up])
        out = self.after_skconv(out)
        return out

class GT_U_DCNet(nn.Module):
    def __init__(self, img_ch  =  1, output_ch  =  1, 
                middle_channel = [64, 128, 256, 512, 1024], 
                encode_len = 5, 
                need_return_dict = True,
                need_supervision = False,
                decode_type = "conv",

        ):
        super(GT_U_DCNet,self).__init__()
        
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
        self.select  =  nn.Conv2d( 2, output_ch, kernel_size = 1, stride = 1, padding = 0)
        middle_channel = middle_channel[ len(middle_channel) - encode_len : ]
        middle_channel =  [ img_ch, *middle_channel]
        self.need_supervision = need_supervision
        print( middle_channel )
        self.pre_encode = _make_bot_layer(
            ch_in = middle_channel[0],         
            ch_out = middle_channel[ 1 ], 
            w = 4
        )
        for i in range(1, encode_len):
            self.encode_list.append( 
                _make_bot_layer(
                    ch_in = 2 * middle_channel[i],             
                    ch_out = middle_channel[ i+1 ], 
                    w = 4
                ) 
            )
        
        for i in range(1, encode_len):
            now_dim = encode_len - i + 1
            next_dim = encode_len - i
            # print( middle_channel[ now_dim ],   middle_channel[ next_dim ])
            self.up_list.append( up_conv(ch_in = middle_channel[now_dim ] , ch_out = 2 * middle_channel[next_dim]) )
            if decode_type == "conv":
                self.decode_list.append( 
                    Decode(ch_in = 2 * middle_channel[ now_dim ], ch_out = middle_channel[next_dim]) 
                    # DSK( middle_channel[ now_dim ], middle_channel[next_dim])
                )
            else:
                self.decode_list.append( 
                     _make_bot_layer(
                        ch_in = 2 * middle_channel[ now_dim ], 
                        ch_out = middle_channel[next_dim],
                        w = 2 ** i
                    ) 
                )
            
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
        self.supervision_list.append( nn.Conv2d(middle_channel[next_dim], output_ch, 1, 1) )


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
    
    def edge_hot_map(self, x):
        x = x.clone().detach()
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
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
            # print( "x{}:{} {}".format( index + 1, out.shape, out_pool.shape ))
            
        x_temp = out_pool 
        # decoding
        for index in range(self.index_len):
            # up = self.up_list[index](x_temp)
            # x_temp = torch.cat( [up, out_list[ self.index_len - index ] ], dim = 1)
            up = out_list[ self.index_len - index ]
            x_temp = self.decode_list[index](up, x_temp)
            # x_temp = self.decode_list[index](x_temp)
            supervision.append(x_temp)
            # self.need_supervision[index](x_temp)
            # print( "decode{}:{} {}".format( index, up.shape, x_temp.shape))
        
        # print("final decode:", x_temp.shape)
        outp = self.last_up(x_temp)
        out = self.last_decode(outp)
        
        edge = self.edge_hot_map(out)
        outp= self.select(torch.cat([outp, edge], 1))
        # print("outp:{}".format( outp.shape ))
        if self.need_supervision:
            for i in range( self.index_len ):
                supervision[i] = self.supervision_list[i](supervision[i])
            return self.build_results(outp, outp, edge, supervision) if self.need_return_dict else( out, edge, supervision) 
        return self.build_results(out, outp, edge) if self.need_return_dict else( out, edge ) 

class GT_U_DCNet_old2(nn.Module):
    def __init__(self, img_ch  =  1, output_ch  =  1, 
                middle_channel = [32, 64, 128, 256, 512], 
                encode_len = 5, 
                need_return_dict = True
        ):
        super(GT_U_DCNet,self).__init__()
        
        self.index_len = encode_len - 1
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2)
        self.Maxpool  =  nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        self.encode_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.decode_list = nn.ModuleList()
        self.select  =  nn.Conv2d( 2, output_ch, kernel_size = 1, stride = 1, padding = 0)
        middle_channel = middle_channel[ len(middle_channel) - encode_len : ]
        middle_channel =  [ img_ch, *middle_channel]
        print( middle_channel )
        self.pre_encode = _make_bot_layer(
            ch_in = middle_channel[0],         
            ch_out = middle_channel[ 1 ], 
            w = 16
        )
        for i in range(1, encode_len):
            self.encode_list.append( 
                _make_bot_layer(
                    ch_in = 2 * middle_channel[i],             
                    ch_out = middle_channel[ i+1 ], 
                    w = 2 ** (encode_len - i)
                ) 
            )
        
        for i in range(1, encode_len):
            now_dim = encode_len - i + 1
            next_dim = encode_len - i
            # print( middle_channel[ now_dim ],   middle_channel[ next_dim ])
            self.up_list.append( up_conv(ch_in = middle_channel[now_dim ] , ch_out = 2 * middle_channel[next_dim]) )
            self.decode_list.append( Decode(ch_in = 2 * middle_channel[ now_dim ], ch_out = middle_channel[next_dim]) )
#                 _make_bot_layer(
#                     ch_in = 2 * middle_channel[ now_dim ], 
#                     ch_out = middle_channel[next_dim],
#                     w = 2 ** i) 
#                     )
            
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
            # print( "x{}:{} {}".format( index + 1, out.shape, out_pool.shape ))
            
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

        edge = self.edge_hot_map(outp)
        outp= self.select(torch.cat([outp, edge], 1))
        # print("outp:{}".format( outp.shape ))
        return self.build_results(outp, outp, edge) if self.need_return_dict else( out, edge ) 


from .pvtv2 import pvt_v2_b2
class GT_U_DC_PVTNet(nn.Module):
    def __init__(self, img_ch  =  1, output_ch  =  1, 
                middle_channel = [64, 128, 256, 512, 1024], 
                encode_len = 5, 
                need_return_dict = True,
                need_supervision = False,
                decode_type = "conv",
                path = '../model/GTU/models/pretrained_pth/pvt_v2_b2.pth'

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
        # self.pre_encode = _make_bot_layer(
        #     ch_in = middle_channel[0],         
        #     ch_out = middle_channel[ 1 ], 
        #     w = 4
        # )
        for i in range(1, encode_len):
            self.encode_list.append( 
                ResGroupFormer( ch_in = 2 * middle_channel[i],             
                    ch_out = middle_channel[ i+1 ],  )
                # nn.Sequential(
                # _make_bot_layer(
                #     ch_in = 2 * middle_channel[i],             
                #     ch_out = middle_channel[ i+1 ], 
                #     w = 4
                # ) ,
                # _make_bot_layer(
                #     ch_in = middle_channel[ i+1 ],             
                #     ch_out = middle_channel[ i+1 ], 
                #     w = 4
                # ) ,
                # _make_bot_layer(
                #     ch_in = middle_channel[ i+1 ],             
                #     ch_out = middle_channel[ i+1 ], 
                #     w = 4
                # ) ,
                # )
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
        # self.edge_hot = nn.Sequential(
            
        # )

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        n_p = sum(x.numel() for x in self.backbone.parameters()) # number parameters
        n_g = sum(x.numel() for x in self.backbone.parameters() if x.requires_grad)  # number gradients
        print(f"pvt Summary: {len(list(self.backbone.modules()))} layers, {n_p} parameters, {n_p/1e6} M, {n_g} gradients")



    def build_results(self, x, y,z, super_vision  = None):
        if super_vision is None:
            return {
            "mask": x,
            "levelset": y,
            "edge":z,
        }
        else:
            return {
                "mask": x,
                "levelset": y,
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
        x[x < 0.5] = 0
        x[x >= 0.5] = 1
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        # edge = self.edge_hot(edge)
        return edge
    
    # @torch.no_grad()
    def pvt_backbone(self, x):
        x = x.clone().detach()
        pvt_x = torch.cat([x,x,x], 1)
        pvt = self.backbone(pvt_x)
        return pvt
    
    def forward(self,x):
        pvt = self.pvt_backbone(x)
        c1, c2, c3, c4 = pvt
        # pvt_decode = [c3, c2, c1]
        pvt_decode = list( reversed( pvt[:len(self.sk_list)] ) )
        
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



class GT_U_DCNet_old(nn.Module):
    def __init__(self, img_ch  =  3,output_ch  =  1, need_return_dict = True):
        super(GT_U_DCNet,self).__init__()
        self.need_return_dict = need_return_dict
        self.Maxpool  =  nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)
        self.Conv1  =  _make_bot_layer(ch_in = img_ch, ch_out = 64)
        self.Conv2  =  _make_bot_layer(ch_in = 64, ch_out = 128)
        self.Conv3  =  _make_bot_layer(ch_in = 128, ch_out = 256)
        self.Conv4  =  _make_bot_layer(ch_in = 256, ch_out = 512)
        self.Conv5  =  _make_bot_layer(ch_in = 512, ch_out = 1024)

        self.Up5  =  up_conv(ch_in = 1024, ch_out = 512)
        self.Up_conv5  =  Decode(ch_in = 1024, ch_out = 512)

        self.Up4  =  up_conv(ch_in = 512, ch_out = 256)
        self.Up_conv4  =  Decode(ch_in = 512, ch_out = 256)
        
        self.Up3  =  up_conv(ch_in = 256, ch_out = 128)
        self.Up_conv3  =  Decode(ch_in = 256, ch_out = 128)
        
        self.Up2  =  up_conv(ch_in = 128, ch_out = 64)
        self.Up_conv2  =  Decode(ch_in = 128, ch_out = 64)

        self.Conv_1x1  =  nn.Conv2d(64,output_ch,kernel_size = 1,stride = 1,padding = 0)
        self.select  =  nn.Conv2d( 2, output_ch, kernel_size = 1, stride = 1, padding = 0)


    def build_results(self, x,y,z):
        return {
            "mask": x,
            "cmask": y,
            "edge":z,
        }
    
    def edge_hot_map(self, x):
        # x = x.clone().detach()
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return edge
    

    def forward(self,x):
        # encoding path
        x1  =  self.Conv1(x)

        x2  =  self.Maxpool(x1)
        x2  =  self.Conv2(x2)
        
        x3  =  self.Maxpool(x2)
        x3  =  self.Conv3(x3)

        x4  =  self.Maxpool(x3)
        x4  =  self.Conv4(x4)

        x5  =  self.Maxpool(x4)
        x5  =  self.Conv5(x5)

        # decoding + concat path
        d5  =  self.Up5(x5)
        d5  =  torch.cat((x4,d5),dim = 1)
        
        d5  =  self.Up_conv5(d5)
        
        d4  =  self.Up4(d5)
        d4  =  torch.cat((x3,d4),dim = 1)
        d4  =  self.Up_conv4(d4)

        d3  =  self.Up3(d4)
        d3  =  torch.cat((x2,d3),dim = 1)
        d3  =  self.Up_conv3(d3)

        d2  =  self.Up2(d3)
        d2  =  torch.cat((x1,d2),dim = 1)
        d2  =  self.Up_conv2(d2)

        d1  =  self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        edge = self.edge_hot_map(d1)
        out = self.select(torch.cat([d1 - edge, edge], 1))
        
        return self.build_results(d1, out, edge) if self.need_return_dict else( d1, out, edge ) 