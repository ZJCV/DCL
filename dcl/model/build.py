# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:20
@file: build.py
@author: zj
@description: 
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from zcls.model.norm_helper import convert_sync_bn
from zcls.util.checkpoint import CheckPointer
from zcls.util import logging
import zcls.util.distributed as du

logger = logging.get_logger(__name__)

from .resnet.resnet import get_resnet
from .mobilenet.mobilenet_v2 import get_mobilenet_v2
from .dclnet.dclnet import DCLNet


def build_model(cfg, device):
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    arch_name = cfg.MODEL.RECOGNIZER.NAME

    if 'resnet' in arch_name or 'resnext' in arch_name:
        model = get_resnet(num_classes=num_classes, arch=arch_name)
    elif 'mobilenet' in arch_name:
        model = get_mobilenet_v2(num_classes=num_classes, arch=arch_name)
    else:
        raise ValueError(f"{arch_name} doesn't exists")

    preloaded = cfg.MODEL.RECOGNIZER.PRELOADED
    if preloaded != "":
        logger.info(f'load preloaded: {preloaded}')
        cpu_device = torch.device('cpu')
        check_pointer = CheckPointer(model)
        check_pointer.load(preloaded, map_location=cpu_device)
        logger.info("finish loading model weights")

    world_size = du.get_world_size()
    if cfg.MODEL.NORM.SYNC_BN and world_size > 1:
        logger.info("start sync BN on the process group of {}".format(du.LOCAL_RANK_GROUP))
        convert_sync_bn(model, du.LOCAL_PROCESS_GROUP)

    if cfg.DCL.USE_DCL:
        model = DCLNet(model, swap_size=cfg.DCL.SWAP_SIZE)

    if du.is_master_proc():
        logger.info(f'full model info:')
        logger.info(model)

    model = model.to(device=device)
    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

    return model
