# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# GrootVL: Tree Topology is All You Need in State Space Model
# Modified by Yicheng Xiao
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from .tree_scanning import Tree_SSM
from fvcore.nn import flop_count
import copy
from functools import partial
from .box_coder import *
from .depatch_embed import Simple_DePatch
from einops import rearrange


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    norm_layer = norm_layer.upper()
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class StemLayer(nn.Module):
    r""" Stem layer of GrootV
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=80,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # print("Stem output:", x.shape)  # 应输出 [2,56,56,80]
        return x


class DownsampleLayer(nn.Module):
    r""" Downsample layer of GrootV
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN', img_size=224):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')
        post_conv_size = img_size // 2
        self.img_size = img_size  # 卷积后的尺寸
        patch_size = 2  # 每个 patch 的大小
        box_coder = pointwhCoder(input_size=self.img_size, patch_count=self.img_size // patch_size,
                             weights=(1., 1., 1., 1.), pts=3, tanh=True,
                             wh_bias=torch.tensor(5. / 3.).sqrt().log())
        self.deform_patch_embed = Simple_DePatch(box_coder, img_size=self.img_size, patch_size=patch_size, patch_pixel=3,
                                             patch_count=self.img_size // patch_size,
                                             in_chans=channels, embed_dim=2*channels, another_linear=True,
                                             use_GE=True,
                                             with_norm=True)
    def forward(self, x):
        # x = self.conv(x.permute(0, 3, 1, 2))
        # x = self.norm(x)
        # print(x.shape)
        # print("DownSample output:", x.shape)  # 应输出 [2,56,56,80]
        # self.img_size=self.img_size//2
        x = rearrange(x, 'b h w d -> b d h w').contiguous()  # 调整输入形状以适配可变形 Patch Embedding
        x = self.deform_patch_embed(x)  # 使用可变形 Patch Embedding 进行下采样
        x = rearrange(x, 'b d h w -> b h w d')  # 将输出形状调整回 (B, H, W, C)
        return x


class MLPLayer(nn.Module):
    r""" MLP layer of GrootV
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GrootVLayer(nn.Module):

    def __init__(self,
                 channels,
                 mlp_ratio=4.,
                 drop=0.,
                 norm_layer='LN',
                 drop_path=0.,
                 act_layer='GELU',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 ):
        super().__init__()
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.TreeSSM = Tree_SSM(
                d_model=channels, 
                d_state=1, 
                ssm_ratio=2,
                ssm_rank_ratio=2,
                dt_rank='auto',
                act_layer=nn.SiLU,
                # ==========================
                d_conv=3,
                conv_bias=False,
                # ==========================
                dropout=0.0,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.TreeSSM(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.TreeSSM(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.TreeSSM(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.TreeSSM(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class GrootVBlock(nn.Module):

    def __init__(self,
                 channels,
                 depth,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            GrootVLayer(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                with_cp=with_cp,
        ) for i in range(depth)
        ])
        self.norm = build_norm_layer(channels, 'LN')
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x

class GrootV(nn.Module):

    def __init__(self,
                 channels=64,
                 depths=[3, 4, 18, 5],
                 num_classes=1000,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 post_norm=False,
                 with_cp=False,
                 cls_scale=1.5,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2**(self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.dims = [channels * (2 ** i) for i in range(len(depths))]
        print(f'using core type: tree_scanning_algorithm')
        print(f'using activation layer: {act_layer}')
        print(f'using main norm layer: {norm_layer}')
        print(f'using dpr: {drop_path_type}, {drop_path_rate}')

        in_chans = 3
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = GrootVBlock(
                channels=self.dims[i],
                depth=depths[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                with_cp=with_cp,
            )
            self.levels.append(level)
        
        self.conv_head = nn.Sequential(
            nn.Conv2d(self.num_features,
                          int(self.num_features * cls_scale),
                          kernel_size=1,
                          bias=False),
        build_norm_layer(int(self.num_features * cls_scale), 'BN',
                                 'channels_first', 'channels_first'),
        build_act_layer(act_layer))
        self.head = nn.Linear(int(self.num_features * cls_scale), num_classes) if num_classes > 0 else nn.Identity()


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_layers = len(depths)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = 'levels.{}.blocks.{}.'.format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios['levels.0.blocks.0.']
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios['levels.1.blocks.0.']
        lr_ratios["levels.0.norm"] = lr_ratios['levels.1.blocks.0.']
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios['levels.2.blocks.0.']
        lr_ratios["levels.1.norm"] = lr_ratios['levels.2.blocks.0.']
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios['levels.3.blocks.0.']
        lr_ratios["levels.2.norm"] = lr_ratios['levels.3.blocks.0.']
        return lr_ratios

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)
        print("conv_head:", x.shape)
        x = self.conv_head(x.permute(0, 3, 1, 2))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_features_seq_out(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level in self.levels:
            x, x_ = level(x, return_wo_downsample=True)
            seq_out.append(x_)
        return seq_out
        
    def forward(self, x):
        # for GrootV-T/S/B/L/XL
        print("forward:", x.shape)
        x = self.forward_features(x)
        print("forward_features:",x.shape)
        x = self.head(x)
        return x

class Backbone_GrootV(GrootV):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.out_indices = out_indices
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])  # 确定通道顺序

        # 为每个输出阶段添加对应的归一化层
        for i in out_indices:
            dim = self.dims[i]
            # 构建归一化层，处理通道顺序转换
            layer = build_norm_layer(
                dim=dim,
                norm_layer=norm_layer,
                in_format='channels_last',  # 输入为channels_last格式
                out_format='channels_first' if self.channel_first else 'channels_last',
                eps=1e-6
            )
            self.add_module(f'outnorm{i}', layer)

        # 删除分类器部分
        del self.head
        del self.conv_head
        del self.avgpool

        # 加载预训练权重
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatible_keys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatible_keys)
        except Exception as e:
            print(f"Failed loading checkpoint from {ckpt}: {e}")

    def forward(self, x):
        # 获取各阶段的特征（channels_last格式）
        seq_out = self.forward_features_seq_out(x)
        outputs = []
        for i, out in enumerate(seq_out):
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(out)  # 应用归一化层（自动处理通道顺序）
                # 确保输出为channels_first格式
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outputs.append(out.contiguous())
        return outputs