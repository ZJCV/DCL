# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午4:00
@file: build.py
@author: zj
@description: 
"""

from .datasets.build import build_dataset
from .transforms.build import build_transform
from .dataloader.build import build_dataloader


def build_data(cfg, is_train=True, **kwargs):
    use_dcl = cfg.DCL.USE_DCL
    swap_size = cfg.DCL.SWAP_SIZE

    transform, target_transform, dcl_transform = build_transform(cfg, is_train=is_train, use_dcl=use_dcl)
    dataset = build_dataset(cfg, transform=transform, target_transform=target_transform, is_train=is_train,
                            dcl_transform=dcl_transform, use_dcl=use_dcl, swap_size=swap_size, **kwargs)

    return build_dataloader(cfg, dataset, is_train=is_train, use_dcl=use_dcl)
