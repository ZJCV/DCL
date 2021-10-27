# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 上午10:04
@file: dclnet.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import ConvBNReLU

from ..resnet.resnet import ResNet
from ..mobilenet.mobilenet_v2 import MobileNetV2

from zcls.config.key_word import KEY_OUTPUT
from dcl.config.key_word import KEY_BACKBONE_FEAT, KEY_CLASSIFIER_FEAT, KEY_OUTPUT_CL, KEY_OUTPUT_DL


class DCLNet(nn.Module):

    def __init__(self, model, swap_size=(7, 7)):
        super().__init__()
        assert isinstance(model, ResNet) or isinstance(model, MobileNetV2)

        feature_dims = model.get_backbone_dims()
        # Adversarial Learning
        # only use cls_2, not realize cls_2xmul
        self.dl_net = nn.Linear(feature_dims, 2, bias=True)
        # Region Alignment Network
        self.cl_net_feature_extractor = nn.Sequential(
            # dw
            ConvBNReLU(feature_dims, feature_dims, stride=2, groups=feature_dims, norm_layer=nn.BatchNorm2d),
            # pw-linear
            nn.Conv2d(feature_dims, feature_dims, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(feature_dims)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        swap_dims = swap_size[0] * swap_size[1]
        self.cl_net_classifier = nn.Linear(feature_dims, swap_dims, bias=True)

        self.model = model

    def forward(self, x):
        res_dict = self.model(x)

        feat1 = res_dict[KEY_BACKBONE_FEAT]
        feat2 = res_dict[KEY_CLASSIFIER_FEAT]

        dl_res = self.dl_net(feat2)

        mask = self.cl_net_feature_extractor(feat1)
        mask = self.avgpool(mask)
        mask = mask.view(mask.size()[0], -1)
        cl_res = self.cl_net_classifier(mask)

        return {
            KEY_OUTPUT: res_dict[KEY_OUTPUT],
            KEY_OUTPUT_DL: dl_res,
            KEY_OUTPUT_CL: cl_res
        }
