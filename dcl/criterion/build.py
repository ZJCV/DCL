# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 上午10:33
@file: build.py
@author: zj
@description: 
"""

from zcls.model import registry
from zcls.model.criterions.crossentropy_loss import CrossEntropyLoss
from zcls.model.criterions.label_smoothing_loss import LabelSmoothingLoss

from .dcl_loss import DCLLoss


def build_criterion(cfg, device):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg).to(device=device)
