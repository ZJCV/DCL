# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 上午11:49
@file: collate.py
@author: zj
@description: 
"""

import torch
import numpy as np

from zcls.config.key_word import KEY_OUTPUT
from dcl.config.key_word import KEY_OUTPUT_CL, KEY_OUTPUT_DL


def collate_fn4train(batch):
    images = []
    label = []
    label_cl = []
    label_dl = []
    for sample in batch:
        images.append(sample[0])
        images.append(sample[1])

        label.append(sample[2])
        label.append(sample[2])

        label_dl.append(1)
        label_dl.append(0)

        label_cl.append(sample[4])
        label_cl.append(sample[5])
    return torch.stack(images, 0), {
        KEY_OUTPUT: torch.from_numpy(np.array(label)),
        KEY_OUTPUT_DL: torch.from_numpy(np.array(label_dl)),
        KEY_OUTPUT_CL: torch.from_numpy(np.array(label_cl))
    }


def collate_fn4test(batch):
    images = []
    label = []
    for sample in batch:
        images.append(sample[0])
        label.append(sample[2])
    return torch.stack(images, 0), torch.from_numpy(np.array(label)),
