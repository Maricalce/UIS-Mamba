# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .grootv import GrootV, Backbone_GrootV  # 显式导出核心类

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet','GrootV', 'Backbone_GrootV', 'build_model'
]


# classification/models/build.py

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type == "grootv":
        model = GrootV(
            in_chans=config.MODEL.GROOTV.IN_CHANS,
            depths=config.MODEL.GROOTV.DEPTHS,
            num_classes=config.MODEL.NUM_CLASSES,
            mlp_ratio=config.MODEL.GROOTV.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            act_layer=config.MODEL.GROOTV.ACT_LAYER,
            norm_layer=config.MODEL.GROOTV.NORM_LAYER,
            **config.MODEL.GROOTV.get('EXTRA_ARGS', {})
        )
        return model
    return None