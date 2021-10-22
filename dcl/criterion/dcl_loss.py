# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 上午10:33
@file: dcl_loss.py
@author: zj
@description: 
"""

from abc import ABC

import torch.nn as nn
from zcls.model import registry
from zcls.config.key_word import KEY_OUTPUT, KEY_LOSS

from dcl.config.key_word import KEY_OUTPUT_CL, KEY_OUTPUT_DL, KEY_TASK_LOSS, KEY_CL_LOSS, KEY_DL_LOSS


@registry.CRITERION.register('DCLLoss')
class DCLLoss(nn.Module, ABC):

    def __init__(self, cfg):
        super(DCLLoss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(reduction='mean')
        # Adversarial Learning
        self.dl_loss = nn.CrossEntropyLoss(reduction='mean')
        # Region Alignment Network
        self.cl_loss = nn.L1Loss(reduction='mean')

        self.lam = cfg.DCL.LAMBDA
        self.beta = cfg.DCL.BETA
        self.gamma = cfg.DCL.GAMMA

    def __call__(self, output_dict, targets):
        assert isinstance(output_dict, dict) and KEY_OUTPUT in output_dict.keys()
        assert isinstance(targets, dict)
        inputs = output_dict[KEY_OUTPUT]
        task_targets = targets[KEY_OUTPUT]
        task_loss = self.task_loss(inputs, task_targets) * self.lam

        res_dl = output_dict[KEY_OUTPUT_DL]
        dl_targets = targets[KEY_OUTPUT_DL]
        dl_loss = self.dl_loss(res_dl, dl_targets) * self.gamma

        res_cl = output_dict[KEY_OUTPUT_CL]
        cl_targets = targets[KEY_OUTPUT_CL]
        cl_loss = self.cl_loss(res_cl, cl_targets) * self.beta

        return {
            KEY_LOSS: task_loss + cl_loss + dl_loss,
            KEY_TASK_LOSS: task_loss,
            KEY_CL_LOSS: cl_loss,
            KEY_DL_LOSS: dl_loss
        }
