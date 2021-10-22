# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午3:52
@file: build.py
@author: zj
@description: 
"""

import torchvision.transforms.transforms as transforms

import zcls.data.transforms.build as build
from zcls.data.transforms import realization


def parse_transform(cfg, is_train=True):
    methods = cfg.DCL.TRAIN_METHODS if is_train else cfg.DCL.TEST_METHODS
    assert isinstance(methods, tuple)

    keys = realization.__all__
    transforms_dict = realization.__dict__
    aug_list = list()

    for method in methods:
        if method in keys:
            transform = transforms_dict[method]
        else:
            raise ValueError(f'f{method} does not exists')

        if method == 'Normalize':
            mean, std, max_pixel_value, p = cfg.DCL.NORMALIZE
            aug_list.append(transform(mean=mean, std=std, max_pixel_value=max_pixel_value, p=p))
        elif method == 'ToTensor':
            p = cfg.DCL.TO_TENSOR
            aug_list.append(transform(p=p))
        else:
            raise ValueError(f'{method} does not exists')

    return transforms.Compose(aug_list)


def build_transform(cfg, is_train=True, use_dcl=False):
    transform, target_transform = build.build_transform(cfg, is_train=is_train)

    dcl_transform = parse_transform(cfg, is_train=is_train) if use_dcl else None

    return transform, target_transform, dcl_transform
