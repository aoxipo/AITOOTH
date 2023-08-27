import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
device = 'cuda:0'
class DWConv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, padding = None, stride = 1, bn = False, relu = True, bias = False):
        super(DWConv, self).__init__()
        self.inp_dim = inp_dim
        if padding is None:
            padding = (kernel_size-1)//2
        self.depthwise = nn.Conv2d(inp_dim, inp_dim, kernel_size, stride, padding = padding, groups=inp_dim, bias=inp_dim)
        self.pointwise = nn.Conv2d(inp_dim, out_dim, kernel_size=1, groups=1)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True, padding = None, bias = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        if padding is None:
            padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=padding, bias = bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DConv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True, padding = None, bias = True):
        super(DConv, self).__init__()
        self.inp_dim = inp_dim
        if padding is None:
            padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=padding,  dilation = 2, bias = bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CrossResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding = 1, Conv_method = Conv):
        super().__init__()
        self.layer = nn.Sequential(
            Conv_method(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv_method(out_channels, out_channels,kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = Conv_method(in_channels, out_channels,kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        x = inputs.clone().detach()
        out = self.layer(x)
        residual  = self.identity_map(inputs)
        skip = out * residual
        return self.relu(skip)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding = 1, Conv_method = Conv):
        super().__init__()
        self.layer = nn.Sequential(
            Conv_method(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv_method(out_channels, out_channels,kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = Conv_method(in_channels, out_channels,kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        x = inputs.clone().detach()
        out = self.layer(x)
        residual  = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)

class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels),
        )

    def forward(self, inputs):
        return self.layer(inputs)

class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = nn.Sequential(
            ResBlock(in_channels + out_channels, out_channels),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels),
        )
        
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate = 0.2, middle_channel = [64,128,256,512]):
        super().__init__()
        self.encoding_layer1_ = ResBlock(input_channel, middle_channel[0])
        self.encoding_layer2_ = DownSampleConv(middle_channel[0], middle_channel[1])
        self.encoding_layer3_ = DownSampleConv(middle_channel[1], middle_channel[2])
        self.bridge = DownSampleConv(middle_channel[2], middle_channel[3])
        self.decoding_layer3_ = UpSampleConv(middle_channel[3], middle_channel[2])
        self.decoding_layer2_ = UpSampleConv(middle_channel[2], middle_channel[1])
        self.decoding_layer1_ = UpSampleConv(middle_channel[1], middle_channel[0])
        self.output = nn.Conv2d(middle_channel[0], output_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, inputs):
        ###################### Enocoder #########################
        e1 = self.encoding_layer1_(inputs)
        e1 = self.dropout(e1)
        e2 = self.encoding_layer2_(e1)
        e2 = self.dropout(e2)
        e3 = self.encoding_layer3_(e2)
        e3 = self.dropout(e3)
        
        ###################### Bridge #########################
        bridge = self.bridge(e3)
        bridge = self.dropout(bridge)
        
        ###################### Decoder #########################
        d3 = self.decoding_layer3_(bridge, e3)
        d2 = self.decoding_layer2_(d3, e2)
        d1 = self.decoding_layer1_(d2, e1)
        
        ###################### Output #########################
        output = self.output(d1)
        return output

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = ResBlock(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = ResBlock(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = ResBlock(nf, nf)
        self.low3 = ResBlock(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2
    
from .RESUNet import MinPool
class HgDiffusion(nn.Module):
    def __init__(self, nstack, input_channel, output_channel, conv_method = 'Conv', bn=False, increase=0, dropout_rate = 0.2, middle_channel = [32,64,128,256]):
        super().__init__()
        self.nstack = nstack
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.Conv_method = conv_method
        self.conv_type_dict = {
            "DWConv": DWConv,
            "Conv": Conv,
            "DConv": DConv,
        }
        if conv_method == 'DConv':
            pad_fix = 2
        else:
            pad_fix = 1
        self.conv = self.conv_type_dict[self.Conv_method]
        
        self.hgs = nn.ModuleList( [
            nn.Sequential(
                # Hourglass(4, input_channel)
                Generator(input_channel, middle_channel[0], middle_channel = middle_channel),
            ) for i in range(nstack)] 
        )

        self.features = nn.ModuleList( [
            nn.Sequential(
            ResBlock(middle_channel[0], output_channel, padding=pad_fix, Conv_method = self.conv),
            self.conv(output_channel, output_channel, 1, bn=True, relu=True)
        ) for i in range(nstack)] )

        self.outs = nn.ModuleList( [self.conv(output_channel, output_channel, 1, relu=False, bn=False) for i in range(nstack)] )
        self.pool = nn.AvgPool2d( 3, 1, padding = 1) 
        self.merge_features = nn.ModuleList( [self.conv(input_channel, input_channel, padding = pad_fix) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [self.conv(output_channel, output_channel, padding = pad_fix) for i in range(nstack-1)] )
        self.final = nn.Conv2d(nstack * middle_channel[0], 1, 1, 1)
        self.relu = nn.ReLU()
        self.erode = MinPool(2,2,1)
        self.dilate = nn.MaxPool2d(2, stride = 1)

    def forward(self, inputs):
        P,C,W,H = inputs.size()
        if( C == 1 or C == 3):
            x = inputs
        else:
            x = inputs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim

        x_backup = x
        combined_hm_preds = []

        for i in range(self.nstack):
            hg = self.hgs[i](x_backup)
            #print("hg:",hg.size()) 
            feature = self.features[i](hg)
            #print("feature:",feature.size())
            preds = self.outs[i](feature)
            keys = self.pool(preds) 
            #print("preds:", preds.size())
            combined_hm_preds.append(self.relu((preds * hg + feature * hg ) * keys))
            if i < self.nstack - 1:
                x_backup = x_backup + self.merge_preds[i](preds) + self.merge_features[i](feature)
        

        feature = torch.cat(combined_hm_preds,1)
        preds = self.final(feature)
        edge = nn.functional.pad(preds, (1, 0, 1, 0))
        edge = self.dilate(edge) - self.erode(edge)
        return preds, edge

class Critic(nn.Module):
    def __init__(self, in_channels=3):
        super(Critic, self).__init__()

        def critic_block(in_filters, out_filters, normalization=True):
            """Returns layers of each critic block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *critic_block(in_channels, 64, normalization=False),
            *critic_block(64, 128),
            *critic_block(128, 256),
            *critic_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
    def forward(self, img_input):
        output = self.model(img_input)
        return output

class VCAttn(nn.Module):
    """
    参考图案 结构 
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_s = nn.Conv2d(in_channels, 1, 1,1) #conv_source
        self.conv_r = nn.Conv2d(in_channels, 1, 1,1) # conv_reference
        self.conv_rs = nn.Conv2d(1, out_channels, 2,2)
        self.conv_rs_1x1 = nn.Conv2d(1, 1, 2,2)
        self.conv_s_a = nn.Conv2d(in_channels + out_channels, out_channels, 3,1,1)
        self.conv_r_a = nn.Conv2d(in_channels + out_channels, out_channels, 3,1,1)
        self.RELU = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x, y):
        x_idn = x.clone().detach()
        x_q = self.conv_s(x)
        y_k =  self.conv_rs_1x1(self.conv_r(y))
        
        CMat = self.softmax(x_q * y_k)
        #print(CMat.shape, y_k.shape)
        y_s = y_k * CMat
        #print(x_idn.shape, y_s.shape)
        attention_map = torch.cat([x_idn, y_s], dim = 1)
        out_put = x_idn + self.RELU(self.conv_s_a(attention_map) * x_idn + self.conv_r_a(attention_map) * y_s)
        out_put = self.RELU(out_put)
        return out_put


class CWGAN(nn.Module):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10,):
        super().__init__()
        self.display_step = display_step
        
        self.generator = Generator(in_channels, out_channels)
        self.critic = Critic(out_channels)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.recon_criterion = nn.L1Loss()
        self.generator_losses, self.critic_losses  = [],[]
    
    def configure_optimizers(self):
        return [self.optimizer_C, self.optimizer_G]
        
    def generator_step(self, real_images, conditioned_images):
        # WGAN has only a reconstruction loss
        self.optimizer_G.zero_grad()
        fake_images = self.generator(conditioned_images)
        recon_loss = self.recon_criterion(fake_images, real_images)
        recon_loss.backward()
        self.optimizer_G.step()
        
        # Keep track of the average generator loss
        self.generator_losses += [recon_loss.item()]
        
    def critic_step(self, real_images, conditioned_images):
        self.optimizer_C.zero_grad()
        fake_images = self.generator(conditioned_images)
        fake_logits = self.critic(fake_images)
        real_logits = self.critic(real_images)
        
        # Compute the loss for the critic
        loss_C = real_logits.mean() - fake_logits.mean()

        # Compute the gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True)
        alpha = alpha.to(device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
        
        interpolated_logits = self.critic(interpolated)
        
        grad_outputs = torch.ones_like(interpolated_logits, dtype=torch.float32, requires_grad=True)
        gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs,create_graph=True, retain_graph=True)[0]

        
        gradients = gradients.view(len(gradients), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradients_penalty
        
        # Compute the R1 regularization loss
        r1_reg = gradients.pow(2).sum(1).mean()
        loss_C += self.lambda_r1 * r1_reg

        # Backpropagation
        loss_C.backward()
        self.optimizer_C.step()
        self.critic_losses += [loss_C.item()]
        
    def training_step(self, batch, batch_idx):
        real, condition = batch
        self.critic_step(real, condition)
        self.generator_step(real, condition)
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        return gen_mean, crit_mean
    
        if batch_idx %self.display_step==0:
            fake = self.generator(condition).detach()
            torch.save(self.generator.state_dict(), "ResUnet_"+ str(batch_idx) +".pt")
            torch.save(self.critic.state_dict(), "PatchGAN_"+ str(batch_idx) +".pt")
            print(f"Epoch {batch_idx} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")
            display_progress(condition[0], real[0], fake[0], batch_idx)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, condition = batch
        if optimizer_idx == 0:
            self.critic_step(real, condition)
        elif optimizer_idx == 1:
            self.generator_step(real, condition)
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        if self.current_epoch%self.display_step == 0 and batch_idx == 0 and optimizer_idx == 1:
            fake = self.generator(condition).detach()
            torch.save(self.generator.state_dict(), "ResUnet_"+ str(self.current_epoch) +".pt")
            torch.save(self.critic.state_dict(), "PatchGAN_"+ str(self.current_epoch) +".pt")
            print(f"Epoch {self.current_epoch} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")
            #display_progress(condition[0], real[0], fake[0], self.current_epoch)