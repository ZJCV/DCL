# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:20
@file: resnet.py
@author: zj
@description: 
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

from zcls.config.key_word import KEY_OUTPUT
from dcl.config.key_word import KEY_BACKBONE_FEAT, KEY_CLASSIFIER_FEAT


class ResNet(nn.Module):

    def __init__(self, num_classes=1000, arch='resnet50'):
        super().__init__()

        assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        self.model = eval(f'models.{arch}')(pretrained=True)

        self.init_weight(num_classes)

    def init_weight(self, num_classes):
        if num_classes != 1000:
            old_fc = self.model.fc
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.fc = new_fc

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features1 = self.model.layer4(x)

        x = self.model.avgpool(features1)
        features2 = torch.flatten(x, 1)
        res = self.model.fc(features2)

        return res, features1, features2

    def forward(self, x):
        res, features1, features2 = self._forward_impl(x)

        return {
            KEY_OUTPUT: res,
            KEY_BACKBONE_FEAT: features1,
            KEY_CLASSIFIER_FEAT: features2
        }

    def get_backbone_dims(self):
        return self.model.fc.in_features


def get_resnet(num_classes=1000, arch='resnet50'):
    return ResNet(num_classes=num_classes, arch=arch)