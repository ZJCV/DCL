# -*- coding: utf-8 -*-

"""
@date: 2021/10/22 下午5:40
@file: crop.py
@author: zj
@description: 
"""

import cv2

from dcl.data.transforms.swap import swap

if __name__ == '__main__':
    img = cv2.imread('tests/assets/lena.jpg')

    swap_img = swap(img, (7, 7))
    cv2.imwrite('tests/assets/lena_swap.jpg', swap_img)
