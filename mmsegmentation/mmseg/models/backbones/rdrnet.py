# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union, List
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
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
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        b, n, c = x.shape
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

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        # print("tensor_a shape:", agent_tokens.shape)#torch.Size([6, 1, 9, 64])
        # print("tensor_b shape:", k.shape)# torch.Size([6, 1, 64, 64])
        # print("tensor_b shape:", position_bias.shape)  # torch.Size([6, 1, 9, 196])
        x2 = (agent_tokens * self.scale) @ k.transpose(-2, -1)
        # print("tensor_b shape:", x2.shape)#torch.Size([6, 1, 9, 64])
        if position_bias.shape[3] > x2.shape[3]:
            position_bias = position_bias.narrow(3, 0, x2.shape[3])
        else:
            position_bias = position_bias.expand(-1, -1, -1, x2.shape[3])
        # print("111tensor_b shape:", position_bias.shape)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        # print("tensor_a shape:", agent_bias.shape)
        x1 = (q * self.scale) @ agent_tokens.transpose(-2, -1)
        if agent_bias.shape[2] > int(x1.shape[2]):
            agent_bias = agent_bias.narrow(2, 0, x1.shape[2])
        else:
            agent_bias = agent_bias.expand(-1, -1, x1.shape[2], -1)

        # print("tensor_a shape:", x1.shape)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 agent_num=49, attn_type='A'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert attn_type in ['A', 'B']
        if attn_type == 'A':
            self.attn = AgentAttention(
                dim, num_patches,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                agent_num=agent_num)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
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

    def __init__(self, img_size=224, patch_size=16, in_chans=128, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # self.conv = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.inchans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # out_size = (math.ceil(x.shape[-2] * 8), math.ceil(x.shape[-1] * 8))
        # x = resize(
        #     x,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=False)
        # print("PatchEmbed之前的x", x.shape)#([6, 128, 128, 128])
        B, C, H, W = x.shape
        # x = x.narrow(1, 0, self.inchans)
        # print("修改后", x.shape)
        x = self.proj(x).flatten(2).transpose(1, 2)
        # print("修改后", x.shape)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=128, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], attn_type='AAAA'):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        attn_type = 'AAAA' if attn_type is None else attn_type
        # self.transformer_blocks = nn.ModuleList()
        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i - 1) * patch_size),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            # self.transformer_blocks.append(Block(
            #     dim=embed_dims[i], num_patches=num_patches, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
            #     qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            #     norm_layer=norm_layer, sr_ratio=sr_ratios[i] if attn_type[i] == 'B' else int(agent_sr_ratios[i]),
            #     agent_num=int(agent_num[i]), attn_type=attn_type[i]))
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_patches=num_patches, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i] if attn_type[i] == 'B' else int(agent_sr_ratios[i]),
                agent_num=int(agent_num[i]), attn_type=attn_type[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        # print("最开始的x", x.shape)  # 6 128 128 128
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            # print("patch_embed之后的x", x.shape)# 6 4 256

            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            # print("位置编码之后的x", x.shape)#6 1 512
            for blk in block:
                x = blk(x, H, W)
                # print("位置编码之后的x", x.shape)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print("sssssssssssssssssss", x.shape)  # 6  1  512

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        return x

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #
    #     return x


class Conv1x1Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1Layer, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.conv1x1(x)
        return x


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
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Squeeze step: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation step
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()

        # Squeeze step
        y = self.global_avg_pool(x)  # Output shape: (batch_size, num_channels, 1, 1)

        # Excitation step
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Scale
        return x * y


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()

        # 定义卷积层，通常使用 7x7 的卷积核
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上进行全局平均池化和全局最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # 最大池化

        # 将两者在通道维度上连接 (cat)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积层生成注意力图
        attention_map = self.conv(x_cat)

        # 通过 sigmoid 激活函数生成最终的注意力图
        attention_map = self.sigmoid(attention_map)

        # 将注意力图与输入特征图逐元素相乘
        return x * attention_map


class FusionModule(nn.Module):
    def __init__(self, Cm, Cg):  # 128 256
        super(FusionModule, self).__init__()
        # 3x3 和 1x1 卷积层
        self.conv3x3 = nn.Conv2d(Cg, Cg * 2, kernel_size=3, padding=1)
        # self.conv1x1 = nn.Conv2d(Cm, Cm, kernel_size=1)
        # self.conv1x1_2 = nn.Conv2d(Cg, Cm, kernel_size=1)
        # self.conv1x1_3 = nn.Conv2d(Cg*2, Cm, kernel_size=1)
        self.conv1x1 = Block1x1(Cg * 2, Cm)
        self.conv1x1_2 = Block1x1(Cm, Cm)
        self.conv1x1_3 = Block1x1(Cg * 2, Cm)
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Sigmoid 激活
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.depthwise = nn.Conv2d(Cm, Cm, kernel_size=3,
                                   stride=1, padding=1, groups=Cm)
        self.ca = ChannelAttention(128)
        self.sa = SpatialAttentionModule()
        # 批量归一化
        self.bn = nn.BatchNorm2d(Cm)
        self.bn1 = nn.BatchNorm2d(Cm * 2)
        self.bn2 = nn.BatchNorm2d(Cm * 4)
        self.relu = nn.ReLU()
        self.ema = EMA(Cm*2)

    def forward(self, T, F):
        # print("TTTTTTTT", T.shape)#([6, 512])
        # print("FFFFFFFFFFF", F.shape)#([6, 128, 128, 128])
        T = T.view(T.shape[0], T.shape[1], 1, 1).expand(F.shape[0], T.shape[1], F.shape[2], F.shape[3])
        # print("222222222222222ddddddddd", x_d.shape)
        if T.shape[1] > F.shape[1]:
            # T = T.narrow(1, 0, F.shape[1])
            T = self.conv1x1_3(self.relu(self.bn2(T)))
        else:
            T = T.expand(-1, F.shape[1], -1, -1)
        # # 假设输入的 T 形状为 (N, D)
        # N, D = T.shape
        # H_t = W_t = int(D ** 0.5)  # 假设 H_t 和 W_t 满足平方关系
        # Ct = T.size(1) // (H_t * W_t)
        # print("TTTTTTTT", T.shape) #[6, 128, 128, 128]
        # print("FFFFFFFFFFF", F.shape) #[6, 128, 128, 128]
        N, Ct, H_t, W_t = T.shape
        # T1 = self.depthwise(T)
        # #print("TTTTTTTT111", T1.shape)#([6, 128 ,128 ,128])
        # T2 = self.conv1x1(self.relu(T))
        # T2 = self.bn(T2)
        # #T2 = self.softmax(T2)
        # T2 = self.softmax(T2)
        # T = T1*T2
        # #print("tttttt", T.shape)#([6, 128, 128, 128])
        # t3 = self.ca(T)
        # T = T+t3
        # F1 = self.gap(self.relu(F))
        # #print("ffff1111", F1.shape)#([6, 128 , 1 ,1])
        # #F1 = F.interpolate(F1, scale_factor=2, mode='bilinear', align_corners=True)
        # F2 = self.conv3x3(self.relu(F))
        # F2 = self.bn(F2)
        # F2 = self.sigmoid(F2)
        # #print("ffff222222", F2.shape)
        # F = F1*F2
        # #print("ffff", F.shape)
        # f3 = self.sa(F)
        # F = f3+F
        # # 最终融合的输出
        # Z_cwf = torch.cat((F, T), dim=1)
        # Z_cwf = self.conv1x1_2(Z_cwf)
        # #print("最终融合的输出", Z_cwf.shape)
        # return Z_cwf

        # 重新reshape Transformer特征
        T_reshaped = T.view(N, Ct, H_t, W_t)

        # CNN 特征的形状 (N, Cf, Hf, Wf)
        Cf, Hf, Wf = F.size(1), F.size(2), F.size(3)

        # Concatenate CNN 和 Transformer 特征
        G = torch.cat((F, T_reshaped), dim=1)  # G 的形状是 (N, Cf + Ct, Hf, Wf)
        # print("gggggg", G.shape)
        # G1 = self.ema(G)
        # G = G1+G
        # 通过 3x3 卷积，1x1 卷积和 GAP 提取混合特征 M
        M = self.conv3x3(self.relu(self.bn1(G)))
        # print("gggggg", M.shape)
        M = self.conv1x1(self.relu(self.bn2(M)))
        # print("gggggg", M.shape)
        M = self.gap(M)
        # print("gggggg", M.shape)
        M = self.conv1x1_2(self.relu(self.bn(M)))
        # print("gggggg", M.shape)
        M = self.sigmoid(M)

        # 重新resize M 以便与 T_reshaped 和 F 进行逐元素乘法
        M_resized = M.view(N, M.size(1), 1, 1)

        # 逐元素乘法
        T_weighted = T_reshaped * M_resized
        F_weighted = F * M_resized

        # 最终融合的输出
        Z_cwf = self.relu(T_weighted + F_weighted)
        # print("最终融合的输出", Z_cwf.shape)
        return Z_cwf

#双分支融合
class APM(nn.Module):
    def __init__(self, in_channels):
        super(APM, self).__init__()

        # 通道注意力组件
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 空间注意力组件
        self.conv3x3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # 通道注意力
        avg_out = self.global_avg_pool(x)
        max_out = self.global_max_pool(x)
        channel_att = self.conv1x1(avg_out + max_out)
        channel_att = torch.sigmoid(channel_att)
        x_channel_att = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x_channel_att, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel_att, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv3x3(spatial_att)
        spatial_att = torch.sigmoid(spatial_att)
        x_spatial_att = x_channel_att * spatial_att

        return x_spatial_att


class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.apm1 = APM(in_channels)
        self.apm2 = APM(in_channels)
        self.w1 = nn.Parameter(torch.tensor(0.5))  # 第一个APM的权重参数
        self.w2 = nn.Parameter(torch.tensor(0.5))  # 第二个APM的权重参数

    def forward(self, semantic_feature, detail_feature):
        apm1_out = self.apm1(semantic_feature * self.w1 + detail_feature * (1 - self.w1))
        apm2_out = self.apm2(semantic_feature * self.w2 + detail_feature * (1 - self.w2))
        fusion_feature = apm1_out + apm2_out
        return fusion_feature

class MLP1(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, use_cross_kv=False,patch_size=16, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], attn_type='AAAA', num_head=8):
        super().__init__()
        assert out_channels % num_head == 0, \
            f"out_channels ({out_channels}) should be a multiple of num_heads ({num_head})"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_head = num_head
        self.use_cross_kv = use_cross_kv
        self.norm = nn.BatchNorm2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert self.same_in_out_chs, "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.k = nn.Parameter(torch.randn(inter_channels, in_channels, 1, 1) * 0.001)
            self.v = nn.Parameter(torch.randn(out_channels, inter_channels, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _act_sn(self, x):
        x = x.view(-1, self.inter_channels, 0, 0) * (self.inter_channels ** -0.5)
        x = F.softmax(x, dim=1)
        x = x.view(1, -1, 0, 0)
        return x

    def _act_dn(self, x):
        x_shape = x.shape
        h, w = x_shape[2], x_shape[3]
        x = x.view(0, self.num_head, self.inter_channels // self.num_head, -1)
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.view(0, self.inter_channels, h, w)
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(x, self.k, stride=2 if not self.same_in_out_chs else 1, padding=0)
            x = self._act_dn(x)
            x = F.conv2d(x, self.v, stride=1, padding=0)
        else:
            assert (cross_k is not None) and (cross_v is not None), \
                "cross_k and cross_v should not be None when use_cross_kv"
            B = x.shape[0]
            assert B > 0, f"The first dim of x ({B}) should be greater than 0"
            x = x.reshape(1, -1, 0, 0)
            x = F.conv2d(x, cross_k, stride=1, padding=0, groups=B)
            x = self._act_sn(x)
            x = F.conv2d(x, cross_v, stride=1, padding=0, groups=B)
            x = x.reshape(-1, self.in_channels, 0, 0)
        return x
class ExternalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, use_cross_kv=False, patch_size=16, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], attn_type='AAAA', num_head=8):
        super().__init__()
        assert out_channels % num_head == 0, \
            f"out_channels ({out_channels}) should be a multiple of num_heads ({num_head})"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_head = num_head
        self.use_cross_kv = use_cross_kv
        self.norm = nn.BatchNorm2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels
        self.Conv1x1Layer1 = Block1x1(128, 128*6)
        self.Conv1x1Layer2 = Block1x1(6, inter_channels)
        self.Conv1x1Layer3 = Block1x1(144, 128 * 6)
        self.Conv1x1Layer4 = Block1x1(6, in_channels)
        self.Conv1x1Layer41 = Block1x1(1, in_channels)
        self.Conv1x1Layer5 = Block1x1(1, in_channels)
        self.Conv1x1Layer6 = Block1x1(1, 128)

        if use_cross_kv:
            assert self.same_in_out_chs, "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.k = nn.Parameter(torch.randn(inter_channels, in_channels, 1, 1) * 0.001)
            self.v = nn.Parameter(torch.randn(out_channels, inter_channels, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _act_sn(self, x):
        B, C, H, W = x.shape
        # print(x.shape) #([6, 6, 117, 117])#492804  yanzheng  1 1 117 245
        # s = self.inter_channels ** -0.5
        # print(s)
        if x.shape[1] == 6:
            x = self.Conv1x1Layer2(x)
        if x.shape[1] == 1:
            x = self.Conv1x1Layer5(x)
        # x = self.Conv1x1Layer2(x)  # * (self.inter_channels ** -0.5) *0.5 #144
        x = F.softmax(x, dim=1)
        return x

    def _act_dn(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_head, self.inter_channels // self.num_head, -1)
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.view(B, self.inter_channels, H, W)
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(x, self.k, stride=2 if not self.same_in_out_chs else 1, padding=0)
            x = self._act_dn(x)
            x = F.conv2d(x, self.v, stride=1, padding=0)
        else:
            assert (cross_k is not None) and (cross_v is not None), \
                "cross_k and cross_v should not be None when use_cross_kv"
            B, C, H, W = x.shape

            # x = x.reshape(B, C, H, W)
            # print("11111111111", x.shape)
            if B==6:
                x = self.Conv1x1Layer1(x)
            x = F.conv2d(x, cross_k, stride=1, padding=0, groups=B)
            x = self._act_sn(x)
            # print("11111111111", x.shape)#([6, 144, 117, 117]) 验证集 1 128 117 245
            if B == 6:
                x = self.Conv1x1Layer3(x)
            # if B == 1:
            #     x = self.Conv1x1Layer6(x)
                # print("11111111111222222222333333344444", x.shape)  # ([6, 128, 12, 12])
            # print("11111111111222222222", x.shape) #([6, 768, 117, 117])
            # print("111111111112222222223333333", cross_v.shape) #([6, 128, 12, 12])
            x = F.conv2d(x, cross_v, stride=1, padding=0, groups=B)
            #print(x.shape) #([6, 6, 106, 106])
            x = resize(
                x,
                size=(H,W),
                mode='bilinear',
                align_corners=False)
            # print(x.shape)
            if x.shape[1] == 6:
                x = self.Conv1x1Layer4(x)
            if x.shape[1] == 1:
                x = self.Conv1x1Layer41(x)
            x = x.reshape(B, self.in_channels, H, W)
        return x


class EABlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=[1, 2, 4, 8], drop_rate=0., drop_path_rate=0., use_injection=True, use_cross_kv=True, cross_size=12, img_size=224, patch_size=16, in_chans=128, num_classes=1000, embed_dims=[64, 128, 256, 512], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], attn_type='AAAA', num_head=8):
        super().__init__()
        in_channels_h, in_channels_l = in_channels     # 128 256
        out_channels_h, out_channels_l = out_channels  # 128 512
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        self.Conv1x1Layer1 = Block1x1(in_channels_l, 128)
        # Low resolution
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                nn.BatchNorm2d(in_channels_l),
                nn.Conv2d(in_channels_l, out_channels_l, kernel_size=1, stride=2, padding=0)
            )
            self.attn_shortcut_l.apply(self._init_weights_kaiming)
        self.attn_l = PyramidVisionTransformer(
            num_classes=num_classes,
            num_heads=num_heads,
            # mlp_ratio=mlp_ratios,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            depths=depths,
            # norm_cfg=norm_cfg,
            attn_type=attn_type,
            sr_ratios=sr_ratios,
            agent_sr_ratios=agent_sr_ratios,
            num_stages=num_stages,
            agent_num=agent_num,
        )
        self.mlp_l = MLP1(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Compression
        self.compression = nn.Sequential(
            nn.BatchNorm2d(out_channels_l),
            nn.ReLU(),
            nn.Conv2d(out_channels_l, out_channels_h, kernel_size=1)
        )
        self.compression.apply(self._init_weights_kaiming)

        # High resolution
        self.attn_h = ExternalAttention(in_channels_h, in_channels_h, inter_channels=cross_size * cross_size, num_head=num_head, use_cross_kv=use_cross_kv)
        self.mlp_h = MLP1(out_channels_h, drop_rate=drop_rate)
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                nn.BatchNorm2d(out_channels_l),
                nn.AdaptiveMaxPool2d(output_size=(self.cross_size, self.cross_size)),
                nn.Conv2d(out_channels_l, 2 * out_channels_h, kernel_size=1, stride=1, padding=0)
            )
            self.cross_kv.apply(self._init_weights)

        # Injection
        if use_injection:
            self.down = nn.Sequential(
                nn.BatchNorm2d(out_channels_h),
                nn.ReLU(),
                nn.Conv2d(out_channels_h, out_channels_l // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels_l // 2),
                nn.ReLU(),
                nn.Conv2d(out_channels_l // 2, out_channels_l, kernel_size=3, stride=2, padding=1)
            )
            self.down.apply(self._init_weights_kaiming)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_h, x_l):
        #print(x_h.shape) # ([6, 128, 128, 128])
        #print(x_l.shape) # ([6, 256, 16, 16])
        # Low resolution
        short_cut_l = x_l
        if self.proj_flag:
            short_cut_l = self.attn_shortcut_l(x_l)
            # print(short_cut_l.shape)  # ([6, 512, 8, 8])
        # print(x_l.shape)
        x_l = self.Conv1x1Layer1(x_l)
        x_l = resize(
            x_l,
            size=(128, 128),
            mode='bilinear',
            align_corners=False)
        x_l = self.attn_l(x_l)
        # print(x_l.shape)  #([6, 512])
        # print(short_cut_l.shape) #([6, 512, 64, 64])
        # x_ld = self.drop_path(x_l)
        # print(x_ld.shape)
        x_l = x_l.view(x_l.shape[0], x_l.shape[1], 1, 1).expand(short_cut_l.shape[0], x_l.shape[1], short_cut_l.shape[2], short_cut_l.shape[3])
        x_l = short_cut_l + self.drop_path(x_l)
        x_l = x_l + self.drop_path(self.mlp_l(x_l))

        # High resolution
        if self.use_cross_kv:
            cross_kv = self.cross_kv(x_l)
            cross_k, cross_v = torch.split(cross_kv, self.out_channels_h, dim=1)
            x_h = self.attn_h(x_h, cross_k, cross_v)
        else:
            x_h = self.attn_h(x_h)
        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        # Injection
        if self.use_injection:
            # print(x_l.shape) #([6, 512, 64, 64])
            x_h1 = self.down(x_h) # ([6, 512, 32, 32])
            # print(x_h1.shape)# ([6, 512, 32, 32])
            x_h1 = resize(
                x_h1,
                size=(x_l.shape[2], x_l.shape[3]),
                mode='bilinear',
                align_corners=False)
            x_l = x_l + self.drop_path(x_h1)

        return x_h, x_l

@MODELS.register_module()
class RDRNet(BaseModule):
    """RDRNet backbone.

    Args:
        in_channels (int): Number of input image channels. Default: 3
        channels: (int): The base channels of RDRNet. Default: 32
        ppm_channels (int): The channels of PPM module. Default: 128
        num_blocks_per_stage (List[int]): The number of blocks with a
            stride of 1 from stage 2 to stage 6. '[4, 3, [5, 4], [5, 4], [1, 1]]'
            corresponding RDRNet-S-Simple, RDRNet-S and RDRNet-M,
            '[6, 5 [7, 6], [7, 6], [2, 2]]' corresponding RDRNet-L.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True)
        init_cfg (dict, optional): Initialization config dict.
            Default: None
        deploy (bool): Whether in deploy mode. Default: False
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
                 patch_size=16, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], attn_type='AAAA',num_head=8
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

        # stage 1-3
        # self.stem = nn.Sequential(
        #     # stage1
        #     RB(in_channels=in_channels, out_channels=channels, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),
        #
        #     # stage2
        #     RB(in_channels=channels, out_channels=channels, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),
        #     *[RB(in_channels=channels, out_channels=channels, stride=1, norm_cfg=self.norm_cfg, deploy=self.deploy) for
        #       _ in range(self.num_blocks_per_stage[0])],
        #
        #     # stage3
        #     RB(in_channels=channels, out_channels=channels * 2, stride=2, norm_cfg=self.norm_cfg, deploy=self.deploy),
        #     *[RB(in_channels=channels * 2, out_channels=channels * 2, stride=1, norm_cfg=self.norm_cfg,
        #          deploy=self.deploy) for _ in range(self.num_blocks_per_stage[1])],
        # )
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
        # self.Conv1x1Layer1 = Conv1x1Layer(960, 128)
        # self.Conv1x1Layer2 = Conv1x1Layer(960, 128)
        self.Conv1x1Layer1 = Block1x1(960, 128)
        self.Conv1x1Layer2 = Block1x1(640, 128)
        self.Conv1x1Layer3 = Block1x1(512, 128)
        self.Conv1x1Layer4 = Block1x1(256, 128)
        self.Conv1x1Layer5 = Block1x1(512, 256)
        self.Conv1x1Layer6 = Block1x1(128, 512)
        self.ema = EMA(640)
        self.ca = ChannelAttention(960)
        self.fm = FusionModule(128, 256)
        self.bn = nn.BatchNorm2d(640)
        self.bn1 = nn.BatchNorm2d(256)
        self.ff = FeatureFusion(128)
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
        self.trans = PyramidVisionTransformer(
            num_classes=num_classes,
            num_heads=num_heads,
            # mlp_ratio=mlp_ratios,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            depths=depths,
            # norm_cfg=norm_cfg,
            attn_type=attn_type,
            sr_ratios=sr_ratios,
            agent_sr_ratios=agent_sr_ratios,
            num_stages=num_stages,
            agent_num=agent_num,
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
        self.down_4 = ConvModule(
            channels * 2,  # 64
            channels * 4,  # 128
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_5 = ConvModule(
            channels * 4,  # 128
            channels * 8,  # 256
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_6 = ConvModule(
            channels * 8,  # 256
            channels * 16,  # 512
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_7 = ConvModule(
            channels * 16,  # 512
            channels * 32,  # 1024
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

        self.layer4 = EABlock(
            in_channels=[channels * 4, channels * 8],
            out_channels=[channels * 4, channels * 16],
            num_heads=num_heads,
            num_head=8,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=True,
            use_cross_kv=True,
            cross_size=12)
        self.layer5 = EABlock(
            in_channels=[channels * 4, channels * 16],
            out_channels=[channels * 4, channels * 16],
            num_heads=num_heads,
            num_head=8,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=True,
            use_cross_kv=True,
            cross_size=12)
        self.spp = RPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5, norm_cfg=self.norm_cfg, deploy=self.deploy)

        self.kaiming_init()

    def forward(self, x):
        """Forward function."""
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        out_size1 = (math.ceil(x.shape[-2] / 32), math.ceil(x.shape[-1] / 32))
        # stage 1-3
        x = self.stem1(x)
        # print(x.size()) #([6, 32, 512, 512])
        x1 = self.down_3(x)  # ([6, 64, 256, 256])
        x1 = self.down_4(x1)  # ([6, 128, 128, 128])
        # x1_1 = self.down_5(x1)  # ([6, 256, 64, 64])
        # x1_2 = self.down_6(x1_1)  # ([6, 512, 32, 32])
        # x1_3 = self.down_7(x1_2)  # ([6, 1024, 16, 16])
        # print(x1.size())# ([6, 128, 128, 128])
        x = self.stem2(x)
        # print(x.size()) #([6, 32, 256, 256])
        x2 = self.down_3(x)  # ([6, 64, 128, 128])
        # x2 = self.down_4(x2)  # ([6, 128, 64, 64])
        # x2_2 = self.down_5(x2_1)  # ([6, 256, 32, 32])
        # x2_3 = self.down_6(x2_2)  # ([6, 512, 16, 16])
        # print(x2.size()) #([6, 128, 128, 128])
        x = self.stem3(x)
        x3 = x
        # print(x.size())  #([6, 64, 128, 128])
        # x3_1 = self.down_4(x)  # ([6, 128, 64, 64])
        # x3_2 = self.down_5(x3_1)  # ([6, 256, 32, 32])
        # x3_3 = self.down_6(x3_2)  # ([6, 512, 16, 16])
        x_123 = torch.cat((x1, x2, x3), dim=1)
        # print(x_123.size())# 6 256 128 128
        x_123 = self.Conv1x1Layer4(self.relu(self.bn1(x_123)))
        # s_p123 = self.trans(x_123)
        # stage4
        x_s = self.semantic_branch_layers[0](x)
        x_s4 = x_s
        x_s4 = resize(
            x_s4,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # x_sp4 = self.trans(x_s4)
        # print(x_sp4.size())
        x_d = self.detail_branch_layers[0](x)
        # print("111111ddddddddd", x_d.shape)#([6, 64, 128, 128])
        # print("1111111ssssss", x_s.shape)#([6, 128, 64, 64])
        #x_d = self.down_1(self.relu(x_d)) #x_d 6 128 64 64

        comp_c = self.compression_1(self.relu(x_s))#128-64
        x_s = x_s + self.down_1(self.relu(x_d))#64-128  s=2
        x_d = x_d + resize(comp_c,
                           size=out_size,
                           mode='bilinear',
                           align_corners=self.align_corners)
        if self.training:
            temp_context = x_d.clone()
        # print("修改后111111ddddddddd", x_d.shape)  # ([6, 64, 128, 128])
        # print("修改后1111111ssssss", x_s.shape)  # ([6, 128, 64, 64])

        # stage5
        x_s = self.semantic_branch_layers[1](self.relu(x_s))
        x_s5 = self.semantic_branch_layers[1](self.relu(x_s4))
        x_sp5 = resize(
            x_s5,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        x_sp5 = self.Conv1x1Layer4(self.relu(self.bn1(x_sp5)))
        # print(x_s5.size())
        # x_sp5 = self.trans(x_sp5)
        # print(x_sp5.size())

        x_d = self.detail_branch_layers[1](self.relu(x_d))
        # print("22222222ddddddddd", x_d.shape)#([6, 64, 128, 128])
        # print("22222222ssssss", x_s.shape)#([6, 256, 32, 32])
        comp_c = self.compression_2(self.relu(x_s))#256-64
        x_s = x_s + self.down_2(self.relu(x_d))#64-128
        x_d = x_d + resize(comp_c,
                           size=out_size,
                           mode='bilinear',
                           align_corners=self.align_corners)
        # print("修改后22222222ddddddddd", x_d.shape)  # ([6, 64, 128, 128])
        # print("修改后22222222ssssss", x_s.shape)  # ([6, 256, 32, 32])
        # stage6
        x_d = self.detail_branch_layers[2](self.relu(x_d))
        x_s = self.semantic_branch_layers[2](self.relu(x_s))
        x_s6 = self.semantic_branch_layers[2](self.relu(x_s5))
        x_sp6 = self.spp(x_s6)
        x_sp6 = resize(
            x_sp6,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # x_sp6 = self.trans(x_sp6)
        # print(x_sp6.size())
        # print("修改后22222222ddddddddd", x_d.shape)  #([6, 128, 128, 128])
        # print("修改后22222222ssssss", x_s.shape)   #([6, 512, 16, 16])
        x_s1 = self.Conv1x1Layer5(self.relu(x_s))
        x_s1 = resize(
            x_s1,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # x_d1 = self.Conv1x1Layer6(self.relu(x_d))
        # 假设 self.layer4 和 self.layer5 是接收列表输入的模块（比如类似于 EABlock）
        # print("修改后22222222ddddddddd", x_d.shape)  #([6, 128, 128, 128])
        # print("修改后22222222ssssss", x_s.shape)   #([6, 512, 16, 16])
        # print("修改后22222222ssssss", x_s1.shape)  # ([6, 256, 16, 16])
        x_d, x_s = self.layer4(self.relu(x_d), self.relu(x_s1))  # 高分辨率输出 x4_, 低分辨率输出 x4  128 256
        x_d, x_s = self.layer5(self.relu(x_d), self.relu(x_s))  # 高分辨率输出 x5_, 低分辨率输出 x5   128 512

        x_s = self.spp(x_s)  # ([6, 128, 16, 16])
        x_s = resize(
            x_s,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # x_sp7 = x_s
        # print("0000000sssssss", x_s.shape)# ([6, 512, 16, 16])
        # x_s = x_s.narrow(1, 0, 128)
        # print("0000000ddddddddd", x_d.shape)  # ([6, 512])

        # x_d4 = self.Conv1x1Layer1(x_d4)
        # x_d5 = self.Conv1x1Layer1(x_d5)
        # print(x_s4.size())  #([6, 128, 64, 64])
        # x_s4 = self.down_5(x_s4)  # ([6, 256, 32, 32])
        # x_s4_1 = self.down_6(x_s4)  # ([6, 512, 16, 16])
        # print(x_s5.size())  #([6, 256, 32, 32])
        # x_s5_1 = self.down_6(x_s5)  # ([6, 512, 16, 16])
        # print(x_s6.size())  #([6, 128, 16, 16])
        # x_s4 = F.interpolate(x_s4, scale_factor=2, mode='nearest')
        # x_s5 = F.interpolate(x_s5, scale_factor=4, mode='nearest')
        # x_s6 = F.interpolate(x_s6, scale_factor=8, mode='nearest')
        # x1_1 = resize(
        #     x1_1,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x1_2 = resize(
        #     x1_2,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x1_3 = resize(
        #     x1_3,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x2_1 = resize(
        #     x2_1,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x2_2 = resize(
        #     x2_2,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x2_3 = resize(
        #     x2_3,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x3_1 = resize(
        #     x3_1,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x3_2 = resize(
        #     x3_2,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # x3_3 = resize(
        #     x3_3,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        #x_s = torch.cat((x_s4, x_s5, x_s6, x1, x2, x3),dim=1)  # (512++128+1024+512+512) 3200   960 4480
        # x_s2 = self.ema(x_s)
        # print("emaemaemaema", x_s2.shape)
        # x_s2 = self.Conv1x1Layer1(self.relu(x_s2))

        # x_s = x_s4 + x_s5 + x_s6
        # print("0000000ddddddddd1111", x_d1.shape) #[6, 256, 128, 128]

        # x_s2 = self.Conv1x1Layer1(x_s2)
        # x_s1 = self.Conv1x1Layer1(x_s1)
        # x_s = x_s + x_s2
        # x_s = x_s + x_s2
        # x_d1 = F.interpolate(x_d1, scale_factor=2, mode='nearest')
        # print("ssssssss1111", x_s.shape) #[6, 128, 128, 128]
        # x_s = self.trans(x_s)
        # print("111111ddddddddd", x_d.shape)  # ([6, 512])
        # print("2222222222ssssssss", x_s.shape)  # ([6, 128, 128, 128])
        # x_sp = torch.cat((x_sp4 , x_sp5 , x_sp6, s_p123), dim=1)
        # x_spa = self.ema(x_sp)
        # x_spa = self.Conv1x1Layer2(x_spa)
        #x_sp = self.Conv1x1Layer2(x_sp)
        # print(x_s4.size())
        # print(x_sp5.size())
        # print(x_sp6.size())
        # print(x_123.size())
        # x_sp = torch.cat((x_s4 , x_sp5 , x_sp6, x_123, x_sp7), dim=1)
        # x_spa = self.ema(x_sp)
        # x_sp = x_sp*x_spa+x_sp
        # x_sp = self.Conv1x1Layer2(self.relu(self.bn(x_sp)))
        # x_spa = self.Conv1x1Layer2(self.relu(self.bn(x_spa)))
        # print(x_sp.size())
        # x_sp = self.trans(x_sp)
        # x_sp = x_s4 + x_sp5 + x_sp6+x_123
        # x_f = self.fm(x_sp, x_s)
        # x_s = x_s.view(x_s.shape[0], x_s.shape[1], 1, 1).expand(x_d.shape[0], x_s.shape[1], x_d.shape[2], x_d.shape[3])
        # print("222222222222222ddddddddd", x_d.shape)
        # if x_d.shape[1] > x_s.shape[1]:
        # x_s = self.Conv1x1Layer3(self.relu(self.bn(x_s)))
        # else:
        #     x_d = x_d.expand(-1, -1, -1, x_s.shape[3])
        # x_f = x_s+x_d
        # print("33333333333333333fffff", x_f.shape)
        # return (temp_context, x_f) if self.training else x_f

        x_ff = self.ff(x_d, x_s)
        # print("fffff", x_ff.shape)
        # return (temp_context, x_d + x_f) if self.training else x_d + x_f
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
