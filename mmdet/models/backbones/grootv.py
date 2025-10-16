"""
UIS-Mamba: Exploring Mamba for Underwater Instance Segmentation 
via Dynamic Tree Scan and Hidden State Weaken

Paper: UIS-Mamba: Exploring Mamba for Underwater Instance Segmentation 
       via Dynamic Tree Scan and Hidden State Weaken
Authors: Runmin Cong, Zongji Yu, Hao Fang, Haoyan Sun, Sam Kwong
Conference: ACM MM 2025
Published: 05 Jul 2025, Last Modified: 11 Jul 2025

Lab: MVPLab (Professor Runmin Cong)
Institution: Shandong University of Finance and Economics

This file implements the UIS-Mamba backbone network, which integrates:
- Dynamic Tree Scan (DTS): Adaptive patch deformation for underwater scenes
- Hidden State Weaken (HSW): Background suppression mechanism
- Tree-SSM: State Space Model with tree-based scanning
- Optional DePatch: Deformable patch embedding for adaptive downsampling

Based on GrootV architecture with significant modifications for underwater
instance segmentation tasks.

References: WaterMask, GrootV
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from .tree_scanning import Tree_SSM
from fvcore.nn import flop_count
import copy
from functools import partial
from .box_coder import pointwhCoder
from .ms_deform_attn_func import MSDeformAttnFunction
from einops import rearrange
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.utils import _pair as to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES

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
        # print("Stem output:", x.shape)  
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding for DePatch
    """
    def __init__(self, img_size=224, patch_size=16, patch_count=14, in_chans=3, embed_dim=768, with_norm=False):
        super().__init__()  
        patch_stride = img_size // patch_count
        patch_pad = (patch_stride * (patch_count - 1) + patch_size - img_size) // 2
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = patch_count * patch_count
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_pad)
        if with_norm:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if hasattr(self, "norm"):
            x = self.norm(x)
        return x


class Simple_Patch(nn.Module):
    def __init__(self, offset_embed, img_size=224, patch_size=2, patch_pixel=3, patch_count=14, 
                 in_chans=3, embed_dim=192, another_linear=False, use_GE=False, local_feature=False, with_norm=False):
        super().__init__()
        self.H, self.W = patch_count, patch_count
        self.num_patches = patch_count * patch_count
        self.another_linear = another_linear
        if self.another_linear:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, in_chans, embed_dim, with_norm=with_norm)
            self.act = nn.GELU() if use_GE else nn.Identity()
            self.offset_predictor = nn.Linear(embed_dim, offset_embed, bias=False)
        else:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, in_chans, offset_embed)

        self.img_size, self.patch_size, self.patch_pixel, self.patch_count = img_size, patch_size, patch_pixel, patch_count
        self.in_chans, self.embed_dim = in_chans, embed_dim

    def reset_offset(self):
        if self.another_linear:
            nn.init.constant_(self.offset_predictor.weight, 0)
            if hasattr(self.offset_predictor, "bias") and self.offset_predictor.bias is not None:
                nn.init.constant_(self.offset_predictor.bias, 0)
        else:
            nn.init.constant_(self.patch_embed.proj.weight, 0)
            if hasattr(self.patch_embed.proj, "bias") and self.patch_embed.proj.bias is not None:
                nn.init.constant_(self.patch_embed.proj.bias, 0)
        print("Parameter for offsets reseted.")

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        B, C, H, W = x.shape
        img = x
        x = self.patch_embed(x)
        if self.another_linear:
            pred_offset = self.offset_predictor(self.act(x))
        else:
            pred_offset = x.contiguous()
        output_size = (H // self.patch_size, W // self.patch_size)
        return self.get_output(img, pred_offset, img_size=(H, W), output_size=output_size), output_size


class Simple_DePatch(Simple_Patch):
    """DePatch implementation for adaptive downsampling in GrootV"""
    def __init__(self, box_coder, show_dim=4, **kwargs):
        super().__init__(show_dim, **kwargs)
        self.box_coder = box_coder
        self.register_buffer("value_level_start_index", torch.as_tensor([0], dtype=torch.long))
        self.output_proj = nn.Linear(self.in_chans * self.patch_pixel * self.patch_pixel, self.embed_dim)
        if kwargs.get("with_norm", False):
            self.with_norm = True
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.with_norm = False

    def get_output(self, img, pred_offset, img_size, output_size):
        B = img.shape[0]
        value_spatial_shapes = torch.as_tensor(img_size, dtype=torch.long, device=pred_offset.device).view(1, 2)
        num_sample_points = self.patch_pixel * self.patch_pixel * output_size[0] * output_size[1]

        sample_location = self.box_coder(pred_offset, img_size=img_size, output_size=output_size)
        sampling_locations = sample_location.view(B, num_sample_points, 1, 1, 1, 2).to(torch.float)
        attention_weights = torch.ones((B, num_sample_points, 1, 1, 1), device=img.device)
        x = img.view(B, self.in_chans, 1, -1).transpose(1, 3).contiguous()
        output = MSDeformAttnFunction.apply(x, value_spatial_shapes, self.value_level_start_index, sampling_locations, attention_weights, 1)
        # output_proj
        output = output.view(B, output_size[0] * output_size[1], self.in_chans * self.patch_pixel * self.patch_pixel)
        output = self.output_proj(output)
        if self.with_norm:
            output = self.norm(output)
        return output


class DePatchDownsampleLayer(nn.Module):
    """DePatch-based Downsample layer for GrootV
    Uses deformable patch embedding for adaptive downsampling
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
        img_size (int): input feature map size
    """
    def __init__(self, channels, norm_layer='LN', img_size=224):
        super().__init__()
        # Calculate the actual feature map size at this stage
        # DePatch parameters
        patch_size = 2  # downsample by 2x
        patch_pixel = 3  # sample 3x3 points per patch
        out_channels = 2 * channels
        
        # Create box coder for deformable sampling
        # img_size here is the current feature map size, not original image size
        patch_count = img_size // patch_size
        wh_bias = torch.tensor(5./3.).sqrt().log()
        self.box_coder = pointwhCoder(
            input_size=img_size, 
            patch_count=patch_count, 
            weights=(1., 1., 1., 1.), 
            pts=patch_pixel, 
            tanh=True, 
            wh_bias=wh_bias
        )
        
        # Create DePatch module
        self.depatch = Simple_DePatch(
            box_coder=self.box_coder,
            show_dim=4,  # offset dimension (x, y, w, h)
            img_size=img_size,
            patch_size=patch_size,
            patch_pixel=patch_pixel,
            patch_count=patch_count,
            in_chans=channels,
            embed_dim=out_channels,
            another_linear=True,
            use_GE=True,
            local_feature=False,
            with_norm=True
        )
        
        # Reset offset predictor to start from identity
        self.depatch.reset_offset()
        
    def forward(self, x):
        # x is in channels_last format (B, H, W, C)
        # Convert to channels_first for DePatch
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        # Apply DePatch
        x, (H, W) = self.depatch(x)  # returns (B, H*W, C) and (H, W)
        
        # Reshape to spatial format and convert back to channels_last
        B, N, C = x.shape
        x = x.view(B, H, W, C)  # (B, H, W, C) - already in channels_last
        
        return x


class DownsampleLayer(nn.Module):
    """ Downsample layer of GrootV
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
        
    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
       
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
                 use_dynamic_tree=False,
                 use_hsw=False,
                 pool_size=2,
                 hsw_varphi=0.7,
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
                # UIS-Mamba: Dynamic Tree Scan + Hidden State Weaken ==========
                use_dynamic_tree=use_dynamic_tree,
                use_hsw=use_hsw,
                pool_size=pool_size,
                hsw_varphi=hsw_varphi,
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
                 use_depatch=False,
                 img_size=224,
                 use_dynamic_tree=True,
                 use_hsw=False,
                 pool_size=2,
                 hsw_varphi=0.7,
                 ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = False  # Add for compatibility

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
                use_dynamic_tree=use_dynamic_tree,
                use_hsw=use_hsw,
                pool_size=pool_size,
                hsw_varphi=hsw_varphi,
        ) for i in range(depth)
        ])
        self.norm = build_norm_layer(channels, 'LN')
        
        # Choose between DePatch and standard downsampling
        if downsample:
            if use_depatch:
                self.downsample = DePatchDownsampleLayer(
                    channels=channels, norm_layer=norm_layer, img_size=img_size)
            else:
                self.downsample = DownsampleLayer(
                    channels=channels, norm_layer=norm_layer, img_size=img_size)
        else:
            self.downsample = None

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
                 # num_classes=1,
                 num_classes=7,
                #  num_classes=80,
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
                 use_depatch=False,
                 img_size=224,
                 use_dynamic_tree=True,
                 use_hsw=False,
                 pool_size=2,
                 hsw_varphi=0.7,
                 **kwargs):
        """
        UIS-Mamba (Underwater Instance Segmentation with Mamba)
        
        Backbone network with Dynamic Tree Scan (DTS) and Hidden State Weaken (HSW) modules.
        
        Args:
            use_depatch: Use DePatch for adaptive downsampling
            use_dynamic_tree: Enable Dynamic Tree Scan (DTS) for adaptive patch deformation
            use_hsw: Enable Hidden State Weaken (HSW) for background suppression
            pool_size: Pooling size for DTS (reduces computation by pool_size^2, default 2 -> 4x reduction)
            hsw_varphi: Background suppression factor for HSW (default 0.7, experimentally optimal)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2**(self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.dims = [channels * (2 ** i) for i in range(len(depths))]
        self.img_size = img_size
        
        print(f'=' * 80)
        print(f'UIS-Mamba Backbone Configuration:')
        print(f'  Core type: tree_scanning_algorithm')
        print(f'  Activation layer: {act_layer}')
        print(f'  Main norm layer: {norm_layer}')
        print(f'  Drop path: {drop_path_type}, rate={drop_path_rate}')
        print(f'  DePatch downsampling: {use_depatch}')
        print(f'  Dynamic Tree Scan (DTS): {use_dynamic_tree}')
        if use_dynamic_tree:
            print(f'    - Pool size: {pool_size} (computation reduced by {pool_size**2}x)')
        print(f'  Hidden State Weaken (HSW): {use_hsw}')
        if use_hsw:
            print(f'    - Background suppression factor (varphi): {hsw_varphi}')
        print(f'=' * 80)

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
            # Calculate feature map size at this stage
            # After stem (2 conv with stride 2), size is img_size // 4
            # Then each downsample layer divides by 2
            stage_img_size = img_size // (4 * (2 ** i))
            
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
                use_depatch=use_depatch,
                img_size=stage_img_size,
                use_dynamic_tree=use_dynamic_tree,
                use_hsw=use_hsw,
                pool_size=pool_size,
                hsw_varphi=hsw_varphi,
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
        # print("conv_head:", x.shape)
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
        # print("forward:", x.shape)
        x = self.forward_features(x)
        # print("forward_features:",x.shape)
        x = self.head(x)
        return x

class Backbone_GrootV(GrootV):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.out_indices = out_indices
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])  

        
        for i in out_indices:
            dim = self.dims[i]
            
            layer = build_norm_layer(
                dim=dim,
                norm_layer=norm_layer,
                in_format='channels_last',  
                out_format='channels_first' if self.channel_first else 'channels_last',
                eps=1e-6
            )
            self.add_module(f'outnorm{i}', layer)

        
        del self.head
        del self.conv_head
        del self.avgpool

        
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
        
        seq_out = self.forward_features_seq_out(x)
        outputs = []
        for i, out in enumerate(seq_out):
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(out)  
                
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outputs.append(out.contiguous())
        return outputs



@BACKBONES.register_module()
class MM_GrootV(BaseModule, Backbone_GrootV):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_GrootV.__init__(self, *args, **kwargs)






