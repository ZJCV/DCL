# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午5:04
@file: mobilenet_v2.py
@author: zj
@description: 
"""

import torch

from dcl.model.mobilenet.mobilenet_v2 import get_mobilenet_v2

if __name__ == '__main__':
    model = get_mobilenet_v2()
    print(model)

    data = torch.randn(1, 3, 224, 224)
    res_dict = model(data)
    for key, value in res_dict.items():
        print(key, value.shape)
