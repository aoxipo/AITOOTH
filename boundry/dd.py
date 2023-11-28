# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1) # _DEFAULT_SCALE_CLAMP
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Boundary(nn.Module):
    def __init__(self, in_channel = 512):
        super().__init__()
        self.feature_encode = nn.Sequential(
            nn.AdaptiveAvgPool2d(int(np.sqrt(in_channel))),
            nn.Flatten(),
        )
        print("Boundary shape,:", int(np.sqrt(in_channel)))
        in_channel = in_channel * 2
        
        self.linear1 = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, in_channel//2),
            nn.GELU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_channel, in_channel//2),
            nn.GELU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(in_channel//2, 1),
            nn.Sigmoid(),
        )
        self.embed = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, in_channel),
            nn.GELU(),
        )

        self.norm1 = nn.Sequential(
            nn.LayerNorm(in_channel)
        )
        self.norm2 = nn.Sequential(
            nn.LayerNorm(in_channel//2)
        )
        
    def forward(self, x, feature, t = None):# feature shape B * 1024 * 8 * 8 # 10 10
        x_skip = self.linear1(x)
        # print(feature.shape, feature)
        feature = self.feature_encode(feature)
        h = torch.cat([x_skip, feature], 1)
        # print(h.shape, self.embed(t).shape)
        if t is not None:
            h += self.embed(t)
            h = self.norm1(h)
        # print(self.linear2(h).shape, x_skip.shape)
        x = self.norm2(x_skip + self.linear2(h))
        x = self.linear3(x)
        return x



def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainerV1(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.l1_loss = nn.L1Loss()
    def forward(self, x, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.rand_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # print(x_t)
        ans = self.model(x_t, t)
        # print(ans)
        loss = self.l1_loss( ans, noise)
        return loss

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.l1_loss = nn.L1Loss()
        
    def forward(self, x_feature, levelset = None):
        levelset, x_0 = self.model_first(x_feature, levelset)
        
        # time map edge
        t = (x_0.clone().detach() * self.T).squeeze(-1).to(torch.int64) # torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        
        noise = torch.rand_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        x_t = torch.clip(x_t, 0, 1)
        
        ans = self.model_train_extract(x_t, levelset, t)
        loss = self.l1_loss( ans, noise)
        return loss

    # generate gt or just use pre-train
    def model_first(self, x0, levelset = None):
        if levelset is None:
            print("warning use mode to predict")
            out, levelset = self.model.pre(x0)
        B,_,_,_ = levelset.shape
        x_0 = torch.rand((B,1)).to(levelset.device)
        
        for ind in range(B):
            levelset[ind] = levelset[ind]/levelset[ind].max()
            levelset[ind][ levelset[ind] < x_0[ind] ] = 0
        return levelset, x_0
    
    def model_train_extract(self, x, levelset, t):
        levelset_index = self.model( x, levelset, t )
        return levelset_index

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, x_feature, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        
        eps = self.model(x_t, x_feature, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def level_first(self, levelset):
        B,_,_,_ = levelset.shape    
        x_t = torch.ones((B,1)).to(levelset.device)
        for ind in range(B):
            levelset[ind] = levelset[ind]/levelset[ind].max()
            # levelset[ind][ levelset[ind] < x_0[ind] ] = 0
        return levelset, x_t
    
    @torch.no_grad()
    def forward(self, levelset):
        """
        Algorithm 2.
        """
        levelset, x_T = self.level_first(levelset)
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # print(t, x_t.shape, levelset.shape)
            mean, var= self.p_mean_variance( x_t, levelset, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = 0# torch.rand_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            # print(x_t)
            # assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   

def model_sample_extract(model, x_t, t, x_feature, levelset, mask, gt):
    out, levelset, middle_feature = model.pre(x_feature)
    levelset_index = model.bround_ddpm( x_t, middle_feature, t )
    levelset_index = x_t - levelset_index
    edge = levelset - levelset_index * torch.ones_like( levelset )
    edge[ edge > 0 ] = 1 
    edge = model.get_edge(edge)          ## levelset - levelset_index * torch.ones_like( levelset )
    model.select( torch.cat([out, edge], 1) )
    outp = model.select(torch.cat([out, edge], 1))
    return outp
