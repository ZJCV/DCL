# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:20
@file: mobilenet_v2.py
@author: zj
@description: 
"""

import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models

from zcls.config.key_word import KEY_OUTPUT
from dcl.config.key_word import KEY_BACKBONE_FEAT, KEY_CLASSIFIER_FEAT


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, arch='mobilenet_v2'):
        super().__init__()

        assert arch in ['mobilenet_v2']
        self.model = eval(f'models.{arch}')(pretrained=True)

        self.init_weight(num_classes)

    def init_weight(self, num_classes):
        if num_classes != 1000:
            old_fc = self.model.classifier[1]
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.classifier[1] = new_fc

    def _forward_impl(self, x: Tensor):
        feat1 = self.model.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(feat1, (1, 1))
        feat2 = torch.flatten(x, 1)
        res = self.model.classifier(feat2)

        return res, feat1, feat2

    def forward(self, x):
        res, features1, features2 = self._forward_impl(x)

        return {
            KEY_OUTPUT: res,
            KEY_BACKBONE_FEAT: features1,
            KEY_CLASSIFIER_FEAT: features2
        }

    def get_backbone_dims(self):
        return self.model.classifier[1].in_features


def get_mobilenet_v2(num_classes=1000, arch='mobilenet_v2'):
    return MobileNetV2(num_classes=num_classes, arch=arch)


if __name__ == '__main__':
    model = get_mobilenet_v2()
    print(model)

    data = torch.randn(1, 3, 224, 224)
    res_dict = model(data)
    for key, value in res_dict.items():
        print(key, value.shape)
