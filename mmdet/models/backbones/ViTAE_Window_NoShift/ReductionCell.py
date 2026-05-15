# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math

class ReductionCell(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, tokens_type='transformer', kernel_size=3, gamma=False, init_values=1e-4, SE=False, downsample_ratio=2):
        super().__init__()
        self.img_size = img_size
        self.norm1 = norm_layer(in_chans)
        self.embed_dims = embed_dims
        if tokens_type == 'transformer':
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=attn_drop)
            self.attn_type = 'transformer'
        else:
            from .NormalCell import AttentionPerformer
            self.attn = AttentionPerformer(embed_dims, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.attn_type = 'performer'
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dims)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        from .NormalCell import Mlp
        self.mlp = Mlp(in_features=embed_dims, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.PCM = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, kernel_size, downsample_ratio, kernel_size//2, 1, group),
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, 1, group),
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, 1, group),
        )
        self.PRM = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, kernel_size, downsample_ratio, kernel_size//2, 1, group),
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, 1, group),
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, 1, group),
        )
        self.pool = nn.AdaptiveAvgPool2d((img_size//downsample_ratio, img_size//downsample_ratio))
        self.proj = nn.Linear(in_chans, embed_dims)
        if gamma:
            self.gamma1 = nn.Parameter(init_values * torch.ones((embed_dims)),requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((embed_dims)),requires_grad=True)
            self.gamma3 = nn.Parameter(init_values * torch.ones((embed_dims)),requires_grad=True)
        else:
            self.gamma1 = self.gamma2 = self.gamma3 = 1
        from .SELayer import SELayer
        self.SE = SELayer(embed_dims) if SE else nn.Identity()
        self.downsample_ratio = downsample_ratio

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_reshape = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_prm = self.PRM(x_reshape).permute(0, 2, 3, 1).reshape(B, -1, self.embed_dims).contiguous()
        x_pcm = self.PCM(x_reshape).permute(0, 2, 3, 1).reshape(B, -1, self.embed_dims).contiguous()
        x_norm = self.norm1(x).permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_pool = self.pool(x_norm)
        x_proj = self.proj(x_pool.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous())
        
        if self.attn_type == 'transformer':
            x_proj_attn = x_proj.permute(1, 0, 2)
            x_prm_attn = x_prm.permute(1, 0, 2)
            x_attn, _ = self.attn(x_proj_attn, x_prm_attn, x_prm_attn)
            x_attn = x_attn.permute(1, 0, 2)
        else:
            x_attn = self.attn(x_proj)
        x = x_proj + self.drop_path(self.gamma1 * x_attn) + self.drop_path(self.gamma2 * x_pcm)
        x = x + self.drop_path(self.gamma3 * self.mlp(self.norm2(x)))
        x = self.SE(x)
        return x, H//self.downsample_ratio, W//self.downsample_ratio
