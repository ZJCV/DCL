# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午2:21
@file: dcl_dataset.py
@author: zj
@description: 
"""

import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from zcls.data.datasets.util import default_converter
from zcls.data.datasets.evaluator.general_evaluator import GeneralEvaluator

from ..transforms.swap import swap, crop_image


class CUB(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, top_k=(1, 5),
                 dcl_transform=None, use_dcl=False, swap_size=(7, 7)):
        self.data_set = ImageFolder(root)

        self.classes = self.data_set.classes
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._update_evaluator(top_k)

        self.dcl_transform = dcl_transform
        self.use_dcl = use_dcl
        self.swap_size = swap_size
        self.is_train = train

    def __getitem__(self, index: int):
        image, target = self.data_set.__getitem__(index)
        image = default_converter(image, rgb=False)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.use_dcl:
            dl_target = -1
            img_swap, cl_target_src, cl_target_swap = self.get_cl_target(image)

            image = self.dcl_transform(image)
            img_swap = self.dcl_transform(img_swap)
            return image, img_swap, target, dl_target, cl_target_src, cl_target_swap
        else:
            return image, target

    def __len__(self) -> int:
        return self.data_set.__len__()

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

    def get_cl_target(self, image):
        img_unswap_list = crop_image(image, self.swap_size)
        swap_range = self.swap_size[0] * self.swap_size[1]
        cl_target_src = [(i - swap_range // 2) / swap_range for i in range(swap_range)]

        img_swap = swap(image, self.swap_size)
        img_swap_list = crop_image(img_swap, self.swap_size)

        unswap_stats = [np.mean(image_block) for image_block in img_unswap_list]
        swap_stats = [np.mean(image_block) for image_block in img_swap_list]

        cl_target_swap = []
        for swap_im in swap_stats:
            distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
            index = distance.index(min(distance))
            cl_target_swap.append((index - (swap_range // 2)) / swap_range)

        return img_swap, cl_target_src, cl_target_swap
