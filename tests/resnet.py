# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午5:03
@file: resnet.py
@author: zj
@description: 
"""

import torch

from dcl.model.resnet.resnet import get_resnet

if __name__ == '__main__':
    model = get_resnet()
    print(model)

    data = torch.randn(1, 3, 224, 224)
    res_dict = model(data)
    for key, value in res_dict.items():
        print(key, value.shape)
