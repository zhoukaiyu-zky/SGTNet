# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union, List
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from functools import partial

from mmseg.models.utils import RPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, T=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.T = T  # Number of top tokens to select

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        token_weights = torch.softmax(q.mean(dim=-1), dim=-1)
        top_weights, top_indices = token_weights.topk(self.T, dim=-1)
        q_top = torch.gather(q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, C // self.num_heads))
        q_top = torch.mean(q_top, dim=2, keepdim=True)
        q_top1 = q_top.permute(0, 1, 3, 2)

        q_m = torch.mean(q, dim=2, keepdim=True)
        q1 = q_top1*q_m
        q_top_expanded = q_top.expand(-1, -1, q.shape[2], -1)
        q1 = torch.mean(q1, dim=2, keepdim=True)
        q1 = q1.expand(-1, -1, q.shape[2], -1)
        q_top = q1+q_top_expanded

        attn = (q_top @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=3, agent_num=49, downstream_agent_shape=(7, 7), scale=-0.5,**kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** scale


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.kernel_size = kernel_size
        self.agent_num = agent_num

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                             padding=kernel_size // 2, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool_size = pool_size
        self.downstream_agent_shape = downstream_agent_shape
        self.pool = nn.AdaptiveAvgPool2d(output_size=downstream_agent_shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        b, n, c = x.shape
        T = int(n/2)
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]


        assert self.sr_ratio == 1
        downstream_agent_num = self.downstream_agent_shape[0] * self.downstream_agent_shape[1]
        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, downstream_agent_num, num_heads, head_dim).permute(0, 2, 1, 3)


        position_bias1 = nn.functional.interpolate(self.an_bias, size=(H, W), mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)

        position_bias1 = nn.functional.interpolate(position_bias1, size=self.downstream_agent_shape, mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, H * W, downstream_agent_num).permute(0, 2, 1)
        position_bias1 = position_bias1.reshape(1, num_heads, downstream_agent_num, H * W).repeat(b, 1, 1, 1)


        position_bias2 = nn.functional.interpolate((self.ah_bias + self.aw_bias).squeeze(0), size=(H, W),
                                                   mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)

        position_bias2 = nn.functional.interpolate(position_bias2, size=self.downstream_agent_shape, mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, H * W, downstream_agent_num).permute(0, 2, 1)
        position_bias2 = position_bias2.reshape(1, num_heads, downstream_agent_num, H * W).repeat(b, 1, 1, 1)

        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v


        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(H, W), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)

        agent_bias1 = nn.functional.interpolate(agent_bias1, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, H * W, downstream_agent_num).repeat(b, 1, 1, 1)


        agent_bias2 = (self.ha_bias + self.wa_bias).squeeze(0).permute(0, 3, 1, 2)
        agent_bias2 = nn.functional.interpolate(agent_bias2, size=(H, W), mode='bilinear')
        agent_bias2 = agent_bias2.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)

        agent_bias2 = nn.functional.interpolate(agent_bias2, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias2 = agent_bias2.reshape(1, num_heads, H * W, downstream_agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2


        token_weights = torch.softmax(q.mean(dim=-1), dim=-1)
        top_weights, top_indices = token_weights.topk(T, dim=-1)
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, q.size(-1))
        q_top = torch.gather(q, dim=2, index=top_indices_expanded)
        q1 = q_top
        q2 = q1 @ agent_tokens.transpose(-2, -1)
        q2 = resize(
            q2,
            size=(q.shape[2],agent_bias.shape[3]),
            mode='bilinear',
            align_corners=False)
        q2 = q2*self.scale
        q_attn = self.softmax(q2 + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H, W, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 agent_num=49, downstream_agent_shape=(7, 7), kernel_size=3, attn_type='A', scale=-0.5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert attn_type in ['A', 'B']
        if attn_type == 'A':
            self.attn = AgentAttention(
                dim, num_patches,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                agent_num=agent_num, downstream_agent_shape=downstream_agent_shape, kernel_size=kernel_size,
                scale=scale)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_size=16, in_chans=128, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class AgentPVT(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=64, num_classes=19, embed_dims=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.05,
                 attn_drop_rate=0.05, drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 3, 4, 2], sr_ratios=[4, 2, 1, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], downstream_agent_shapes=[(3, 3), (4, 4), (7, 7), (7, 7)],
                 kernel_size=3, attn_type='AAAA', scale=-0.5, init_cfg=None, **kwargs):
        super().__init__()
        self.init_cfg = init_cfg
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        self.agent_num = agent_num
        self.downstream_agent_shapes = downstream_agent_shapes
        self.attn_type = attn_type

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        attn_type = 'AAAA' if attn_type is None else attn_type
        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i - 1) * patch_size),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_patches=num_patches, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i] if attn_type[i] == 'B' else int(agent_sr_ratios[i]),
                agent_num=int(agent_num[i]), downstream_agent_shape=downstream_agent_shapes[i],
                kernel_size=kernel_size, attn_type=attn_type[i], scale=scale)
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)


        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(AgentPVT, self).train(mode)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)  # b*g,c//g,1,w
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # b*g,c//g,1,h
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # b*g,c//g,1,h+w
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class ConvBranch(nn.Module):
    def __init__(self, in_channels):
        super(ConvBranch, self).__init__()
        self.conv1x3 = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1))
        self.conv3x1 = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0))
        self.conv1x5 = nn.Conv2d(in_channels, in_channels, (1, 5), padding=(0, 2))
        self.conv5x1 = nn.Conv2d(in_channels, in_channels, (5, 1), padding=(2, 0))

    def forward(self, x):
        out1 = self.conv1x3(x)
        out2 = self.conv3x1(out1)
        out3 = self.conv1x5(x)
        out4 = self.conv5x1(out3)
        return out2 + out4

class SIEM(nn.Module):
    def __init__(self, in_channels):
        super(SIEM, self).__init__()
        self.Conv1x1Layer = Block1x1(in_channels * 4, in_channels)
        self.Conv1x1Layer1 = Block1x1(in_channels, in_channels * 4,)
        self.conv3x3 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3)
        self.conv_branch = ConvBranch(in_channels)

        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.convout = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.Conv1x1Layer(x1)
        x2 = self.Conv1x1Layer(x2)
        x = torch.concat([x1, x2], dim=1)
        x2 = self.conv_branch(x2)
        x1 = self.gap(x1) + self.gmp(x1)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.gap(x)
        x = self.convout(x)
        x = nn.Sigmoid()(x)
        x = x1*x + x2*x
        x = nn.ReLU6()(x)
        x = self.Conv1x1Layer1(x)
        return x

class EAPM(nn.Module):
    def __init__(self, in_channels):
        super(EAPM, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # b,c,1,w
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # b,c,1,h
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # b,c,1,h+w
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        hw_out = x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        gap = self.global_avg_pool(hw_out)
        map = self.global_max_pool(hw_out)
        channel_att = self.conv1x1_1(gap + map)
        channel_att = channel_att * x
        avg_out = torch.mean(channel_att, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_att, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv3x3_1(spatial_att)
        x_spatial_att1 = channel_att * spatial_att
        x_spatial_att = torch.sigmoid(x_spatial_att1)
        return x_spatial_att


class AGFM(nn.Module):
    def __init__(self, in_channels):
        super(AGFM, self).__init__()
        self.eapm = EAPM(in_channels)
        self.Conv1x1Layer = Block1x1(384, 128)

    def forward(self, semantic_feature, detail_feature):
        s_d = semantic_feature+detail_feature
        w = self.eapm(s_d)
        s1 = w*semantic_feature
        d1 = (1-w)*detail_feature
        s_d1 = s1+d1
        s_d2 = torch.cat((semantic_feature, detail_feature, s_d1), dim=1)
        x_d2 = self.Conv1x1Layer(s_d2)
        fusion_feature = x_d2 + s1+ d1
        return fusion_feature



@MODELS.register_module()
class SGTNet(BaseModule):
    """SGTNet backbone.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List[int] = [4, 3, [5, 4], [5, 4], [1, 1]],
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False,
                 embed_dims=[32, 64, 128, 256],
                 ):
        super().__init__(init_cfg)
        self.embed_dim = embed_dims
        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.deploy = deploy

        # stage 1
        self.stem1 = nn.Sequential(
            RB(in_channels=in_channels, out_channels=channels, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy))

        # stage2
        self.stem2 = nn.Sequential(
            RB(in_channels=channels, out_channels=channels, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),
            *[RB(in_channels=channels, out_channels=channels, stride=1, norm_cfg=self.norm_cfg, deploy=self.deploy) for
              _ in range(self.num_blocks_per_stage[0])])

        # stage3
        self.stem3 = nn.Sequential(
            RB(in_channels=channels, out_channels=channels * 2, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),
            *[RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg,
                 deploy=self.deploy) for _ in range(self.num_blocks_per_stage[1])])
        self.relu = nn.ReLU()
        self.ema = EMA(64)
        self.siem = SIEM(128)
        self.agfm = AGFM(128)
        # semantic branch
        self.semantic_branch_layers = nn.ModuleList()
        self.semantic_branch_layers.append(
            nn.Sequential(
                RB(in_channels=channels * 2, out_channels=channels * 4, stride=2, norm_cfg=self.norm_cfg,
                   deploy=self.deploy),
                *[RB(in_channels=channels * 4, out_channels=channels * 4, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[2][0] - 1)],
                RB(in_channels=channels * 4, out_channels=channels * 4, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.semantic_branch_layers.append(
            nn.Sequential(
                RB(in_channels=channels * 4, out_channels=channels * 8, stride=2, norm_cfg=self.norm_cfg,
                   deploy=self.deploy),
                *[RB(in_channels=channels * 8, out_channels=channels * 8, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[3][0] - 1)],
                RB(in_channels=channels * 8, out_channels=channels * 8, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.trans = AgentPVT(
            img_size=256,
            patch_size=8,
            in_chans=64,
            num_classes=19,
            embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            agent_sr_ratios='1111',
            num_stages=4,
            agent_num=[9, 16, 49, 49],
            downstream_agent_shapes=[(12, 12), (16, 16), (28, 28), (28, 28)],
            kernel_size=3,
            attn_type='AAAA',
            scale=-0.5
        )

        self.semantic_branch_layers.append(
            nn.Sequential(
                self._make_layer(
                    block=Bottleneck,
                    inplanes=channels * 8,
                    planes=channels * 8,
                    num_blocks=self.num_blocks_per_stage[4][0],
                    stride=2),
            )
        )

        # bilateral fusion
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))
        self.down_3 = ConvModule(
            channels,  # 32
            channels * 2,  # 64
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        # detail branch
        self.detail_branch_layers = nn.ModuleList()
        self.detail_branch_layers.append(
            nn.Sequential(
                *[RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[2][1] - 1)],
                RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.detail_branch_layers.append(
            nn.Sequential(
                *[RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg,
                     deploy=self.deploy) for _ in range(self.num_blocks_per_stage[3][1] - 1)],
                RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg, act=False,
                   deploy=self.deploy),
            )
        )
        self.detail_branch_layers.append(
            self._make_layer(
                block=Bottleneck,
                inplanes=channels * 2,
                planes=channels * 2,
                num_blocks=self.num_blocks_per_stage[4][1],
            )
        )

        self.spp = RPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5, norm_cfg=self.norm_cfg, deploy=self.deploy)

        self.kaiming_init()

    def forward(self, x):
        """Forward function."""
        global temp_context
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        # stage 1-3
        x = self.stem1(x)
        x1 = self.down_3(x)  # ([6, 64, 256, 256])
        x = self.stem2(x)
        x = self.stem3(x)
        # stage4
        x_s = self.semantic_branch_layers[0](x)
        x_d = self.detail_branch_layers[0](x)
        comp_c = self.compression_1(self.relu(x_s))#128-64
        x_s = x_s + self.down_1(self.relu(x_d))#64-128  s=2
        x_d = x_d + resize(comp_c,
                           size=out_size,
                           mode='bilinear',
                           align_corners=self.align_corners)
        if self.training:
            temp_context = x_d.clone()
        # stage5
        x_s = self.semantic_branch_layers[1](self.relu(x_s))
        x_d = self.detail_branch_layers[1](self.relu(x_d))
        comp_c = self.compression_2(self.relu(x_s))#256-64
        x_s = x_s + self.down_2(self.relu(x_d))#64-128
        x_d = x_d + resize(comp_c,
                           size=out_size,
                           mode='bilinear',
                           align_corners=self.align_corners)
        # stage6
        x_d = self.detail_branch_layers[2](self.relu(x_d))
        x_s = self.semantic_branch_layers[2](self.relu(x_s))
        x_spa = self.ema(x1)
        x_sp = x_spa+x1
        x_sp1 = self.trans(x_sp)
        x_sp = resize(
            x_sp1,
            size=(x_s.shape[2], x_s.shape[3]),
            mode='bilinear',
            align_corners=self.align_corners)
        x_s = self.siem(x_sp, x_s)
        x_s = self.spp(x_s)
        x_s = resize(
            x_s,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        x_ff = self.agfm(x_d, x_s)
        return (temp_context, x_ff) if self.training else x_ff

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                norm_cfg=self.norm_cfg,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RB):
                m.switch_to_deploy()
        self.spp.switch_to_deploy()
        self.deploy = True

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block1x1(BaseModule):
    """The 1x1 path of the Reparameterizable Block.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            stride (int or tuple): Stride of the convolution. Default: 1
            padding (int, tuple): Padding added to all four sides of
                the input. Default: 1
            bias (bool) : Whether to use bias.
                Default: True
            norm_cfg (dict): Config dict to build norm layer.
                Default: dict(type='BN', requires_grad=True)
            deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 bias: bool = True,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 deploy: bool = False):
        super().__init__()

        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        return x

    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = self.conv2.conv
        self.conv.weight.data = torch.matmul(kernel2.transpose(1, 3), kernel1.squeeze(3).squeeze(2)).transpose(1, 3)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class RB(nn.Module):
    """Reparameterizable Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True)
        act (bool) : Whether to use activation function.
            Default: False
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 act: bool = True,
                 deploy: bool = False):
        super().__init__()

        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if act:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Identity()

        if deploy:
            self.reparam_3x3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                padding_mode=padding_mode)

        else:
            if (out_channels == in_channels) and stride == 1:
                self.path_residual = build_norm_layer(norm_cfg, num_features=in_channels)[1]
            else:
                self.path_residual = None

            self.path_3x3 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.path_1x1 = Block1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding_11,
                bias=True,
                norm_cfg=norm_cfg,
            )

    def forward(self, inputs: Tensor) -> Tensor:

        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(inputs))

        if self.path_residual is None:
            id_out = 0
        else:
            id_out = self.path_residual(inputs)

        return self.relu(self.path_3x3(inputs) + self.path_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.path_3x3)
        self.path_1x1.switch_to_deploy()
        kernel1x1, bias1x1 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kernelid, biasid = self._fuse_bn_tensor(self.path_residual)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific conv layer.

        Args:
            conv (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if conv is None:
            return 0, 0
        if isinstance(conv, ConvModule):
            kernel = conv.conv.weight
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
        else:
            assert isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    conv.weight.device)
            kernel = self.id_tensor
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'reparam_3x3'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(
            in_channels=self.path_3x3.conv.in_channels,
            out_channels=self.path_3x3.conv.out_channels,
            kernel_size=self.path_3x3.conv.kernel_size,
            stride=self.path_3x3.conv.stride,
            padding=self.path_3x3.conv.padding,
            bias=True)
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('path_3x3')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
