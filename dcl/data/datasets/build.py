# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午2:21
@file: build.py
@author: zj
@description: 
"""

from .cifar import CIFAR
from .cub import CUB


def build_dataset(cfg, transform=None, target_transform=None, is_train=True,
                  dcl_transform=None, use_dcl=False, swap_size=(7, 7), **kwargs):
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT
    top_k = cfg.DATASET.TOP_K

    if dataset_name == 'CIFAR100':
        dataset = CIFAR(data_root, train=is_train, transform=transform, target_transform=target_transform,
                        top_k=top_k, is_cifar100=True,
                        dcl_transform=dcl_transform, use_dcl=use_dcl, swap_size=swap_size)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR(data_root, train=is_train, transform=transform, target_transform=target_transform,
                        top_k=top_k, is_cifar100=False,
                        dcl_transform=dcl_transform, use_dcl=use_dcl, swap_size=swap_size)
    elif dataset_name == 'CUB':
        dataset = CUB(data_root, train=is_train, transform=transform, target_transform=target_transform, top_k=top_k,
                      dcl_transform=dcl_transform, use_dcl=use_dcl, swap_size=swap_size)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset
