# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午2:18
@file: swap.py
@author: zj
@description: 
"""

import cv2
import random

import numpy as np


def crop_image(image, crop_nums):
    assert isinstance(image, np.ndarray)
    assert len(crop_nums) == 2

    high, width = image.shape[:2]
    crop_x = [int((width / crop_nums[0]) * i) for i in range(crop_nums[0] + 1)]
    crop_y = [int((high / crop_nums[1]) * i) for i in range(crop_nums[1] + 1)]

    im_list = []
    for j in range(len(crop_y) - 1):
        for i in range(len(crop_x) - 1):
            im_list.append(image[crop_y[j]:min(crop_y[j + 1], high), crop_x[i]:min(crop_x[i + 1], width)])
    return im_list


def swap(image, crop_nums):
    assert isinstance(image, np.ndarray)
    assert len(crop_nums) == 2
    src_width, src_height = image.shape[:2]

    image = image[10:(src_height - 10), 10:(src_width - 10)]
    swap_img_list = crop_image(image, crop_nums)

    tmp_x_list = []
    tmp_y_list = []
    count_x = 0
    count_y = 0
    k = 1
    # swap range
    neighbor = 2
    for i in range(crop_nums[1] * crop_nums[0]):
        tmp_x_list.append(swap_img_list[i])
        count_x += 1
        if len(tmp_x_list) >= k:
            tmp = tmp_x_list[count_x - neighbor:count_x]
            random.shuffle(tmp)
            tmp_x_list[count_x - neighbor:count_x] = tmp
        if count_x == crop_nums[0]:
            tmp_y_list.append(tmp_x_list)
            count_x = 0
            count_y += 1
            tmp_x_list = []
        if len(tmp_y_list) >= k:
            tmp2 = tmp_y_list[count_y - neighbor:count_y]
            random.shuffle(tmp2)
            tmp_y_list[count_y - neighbor:count_y] = tmp2
    random_im = []
    for line in tmp_y_list:
        random_im.extend(line)

    image_height, image_width = image.shape[:2]
    iw = int(image_width / crop_nums[0])
    ih = int(image_height / crop_nums[1])
    dst_image = np.zeros(image.shape).astype(np.uint8)
    x = 0
    y = 0
    for im in random_im:
        dst_image[y * ih:(y + 1) * ih, x * iw:(x + 1) * iw] = cv2.resize(im, (iw, ih))

        x += 1
        if x == crop_nums[0]:
            x = 0
            y += 1

    return cv2.resize(dst_image, (src_width, src_height))
