# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# GrootVL: Tree Topology is All You Need in State Space Model
# Modified by Yicheng Xiao
# --------------------------------------------------------
# classification/models/__init__.py

from .grootv import GrootV, Backbone_GrootV  # 显式导出核心类
from .build import build_model  # 导出模型构建函数

__all__ = ['GrootV', 'Backbone_GrootV', 'build_model']  # 明确导出项

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