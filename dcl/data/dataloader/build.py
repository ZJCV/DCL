# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 上午11:52
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

import zcls.util.distributed as du
from zcls.data.datasets.mp_dataset import MPDataset

from .collate import collate_fn4train, collate_fn4test


def build_sampler(cfg, dataset):
    world_size = du.get_world_size()
    num_gpus = cfg.NUM_GPUS
    rank = du.get_rank()

    if num_gpus <= 1:
        if cfg.DATALOADER.RANDOM_SAMPLE:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    else:
        shuffle = cfg.DATALOADER.RANDOM_SAMPLE
        sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=shuffle)

    return sampler


def build_dataloader(cfg, dataset, is_train=True, use_dcl=False):
    batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE if is_train else cfg.DATALOADER.TEST_BATCH_SIZE

    if use_dcl:
        collate_fn = collate_fn4train if is_train else collate_fn4test
    else:
        collate_fn = None
    sampler = None if isinstance(dataset, MPDataset) else build_sampler(cfg, dataset)
    data_loader = DataLoader(dataset,
                             collate_fn=collate_fn,
                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                             sampler=sampler,
                             batch_size=batch_size,
                             drop_last=is_train,
                             # [When to set pin_memory to true?](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723)
                             pin_memory=True)

    return data_loader
