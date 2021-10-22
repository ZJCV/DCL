# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 上午9:55
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from zcls.config import get_cfg_defaults


def add_custom_config(_C):
    # Add your own customized config.
    _C.DCL = CN()
    _C.DCL.USE_DCL = False

    _C.DCL.SWAP_SIZE = (7, 7)

    _C.DCL.LAMBDA = 1.0
    _C.DCL.BETA = 1.0
    _C.DCL.GAMMA = 1.0

    _C.DCL.TRAIN_METHODS = ('ToTensor', 'Normalize')
    _C.DCL.TEST_METHODS = ('ToTensor', 'Normalize')
    _C.DCL.NORMALIZE = ((0.445, 0.445, 0.445), (0.225, 0.225, 0.225), 255.0, 1.0)
    _C.DCL.TO_TENSOR = 1.0

    return _C


cfg = add_custom_config(get_cfg_defaults())
