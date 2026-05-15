# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
import torch
import torch.nn as nn
from mmdet.utils import get_root_logger
from ...builder import BACKBONES
from .NormalCell import NormalCell
from .ReductionCell import ReductionCell
import math
from timm.models.layers import trunc_normal_
from mmcv.runner import load_checkpoint
import numpy as np

@BACKBONES.register_module()
class ViTAE_Window_NoShift_basic(nn.Module):
    def __init__(self, img_size=224, in_chans=3, stages=4, embed_dims=64, token_dims=64, downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 3, 3, 3], 
                RC_heads=[1, 1, 1, 1], NC_heads=[4, 4, 4, 4], RC_op='transformer', NC_op='transformer', RC_tokens_type=['performer', 'transformer', 'transformer', 'transformer'], NC_tokens_type=['transformer', 'transformer', 'transformer', 'transformer'],
                RC_group=[1, 1, 1, 1], NC_group=[1, 1, 1, 1], NC_depth=[2, 2, 6, 2], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, window_size=7, out_indices=(0, 1, 2, 3), frozen_stages=-1, 
                use_checkpoint=False, norm_eval=False, gamma=False, init_values=1e-4, SE=False, relative_pos=False, pretrained=None, init_cfg=None):
        super().__init__()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.norm_eval = norm_eval
        self.num_stages = stages
        self.RC_layers = nn.ModuleList()
        self.NC_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(NC_depth))]
        
        curr_img_size = img_size
        for i in range(stages):
            if i == 0:
                input_dim = in_chans
            else:
                input_dim = token_dims[i-1]
            
            self.RC_layers.append(
                ReductionCell(img_size=curr_img_size, in_chans=input_dim, embed_dims=embed_dims[i], 
                            num_heads=RC_heads[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                            attn_drop=attn_drop_rate, drop_path=0, group=RC_group[i], tokens_type=RC_tokens_type[i], kernel_size=kernel_size[i],
                            gamma=gamma, init_values=init_values, SE=SE, downsample_ratio=downsample_ratios[i])
            )
            
            curr_img_size = curr_img_size // downsample_ratios[i]

            NC_stage = nn.ModuleList([
                NormalCell(dim=embed_dims[i], num_heads=NC_heads[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                          attn_drop=attn_drop_rate, drop_path=dpr[sum(NC_depth[:i]) + j], group=NC_group[i], tokens_type=NC_tokens_type[i], 
                          window_size=window_size, shift_size=0 if j % 2 == 0 else window_size // 2, img_size=curr_img_size,
                          gamma=gamma, init_values=init_values, SE=SE, relative_pos=relative_pos)
                for j in range(NC_depth[i])
            ])
            self.NC_layers.append(NC_stage)
            
            if i in self.out_indices:
                self.out_layers.append(norm_layer(embed_dims[i]))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        outs = []
        for i in range(self.num_stages):
            x, H, W = self.RC_layers[i](x, H, W)
            for j in range(len(self.NC_layers[i])):
                x, H, W = self.NC_layers[i][j](x, H, W)
            if i in self.out_indices:
                out = self.out_layers[self.out_indices.index(i)](x)
                out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
