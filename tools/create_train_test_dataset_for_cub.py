# -*- coding: utf-8 -*-

"""
@date: 2021/9/23 上午11:18
@file: create_train_test_dataset.py
@author: zj
@description: using CUB's config files to split train/test dataset
"""

import os
import shutil

from tqdm import tqdm

import numpy as np


def parse_txt_path(data_path):
    assert os.path.isfile(data_path)

    data_array = np.loadtxt(data_path, dtype=str, delimiter=' ')
    return list(data_array[:, 1])


def parse(image_path_list, split_list, src_root, dst_root):
    assert os.path.isdir(src_root), src_root
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    assert os.path.isdir(dst_root), dst_root

    train_root = os.path.join(dst_root, 'train')
    assert not os.path.exists(train_root), train_root
    os.makedirs(train_root)

    test_root = os.path.join(dst_root, 'test')
    assert not os.path.exists(test_root), test_root
    os.makedirs(test_root)

    for image_path, flag in tqdm(zip(image_path_list, split_list)):
        cls_name, img_name = os.path.split(image_path)
        if int(flag) == 1:
            cls_dir = os.path.join(train_root, cls_name)
        else:
            cls_dir = os.path.join(test_root, cls_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        assert os.path.isdir(cls_dir), cls_dir

        src_image_path = os.path.join(src_root, image_path)
        assert os.path.isfile(src_image_path), src_image_path
        dst_img_path = os.path.join(cls_dir, img_name)
        shutil.copy(src_image_path, dst_img_path)


if __name__ == '__main__':
    images_path = '/home/zj/data/cub/CUB_200_2011/images.txt'
    image_path_list = parse_txt_path(images_path)

    train_test_split_path = '/home/zj/data/cub/CUB_200_2011/train_test_split.txt'
    split_list = parse_txt_path(train_test_split_path)

    src_root = '/home/zj/data/cub/CUB_200_2011/images'
    dst_root = 'data/cub/'
    parse(image_path_list, split_list, src_root, dst_root)
