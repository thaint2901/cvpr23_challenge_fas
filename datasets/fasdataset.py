# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
Authors: xuyaowen(xuyw1@chinatelecom.cn)
Date:    2023/02/08
"""

import math
import cv2
import numpy as np
import mxnet as mx
import albumentations as A
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchtoolbox.transform import Cutout

def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w/2+x, box_h/2+y

    left_top_x = center_x-new_width/2
    left_top_y = center_y-new_height/2
    right_bottom_x = center_x+new_width/2
    right_bottom_y = center_y+new_height/2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w-1:
        left_top_x -= right_bottom_x-src_w+1
        right_bottom_x = src_w-1

    if right_bottom_y > src_h-1:
        left_top_y -= right_bottom_y-src_h+1
        right_bottom_y = src_h-1

    return int(left_top_x), int(left_top_y),\
            int(right_bottom_x), int(right_bottom_y)

def get_gaussian_band_pass_filter(shape, cutoff_high = 2, cutoff_low = 30):
    """
        Gaussian band pass filter
    """
    d0 = cutoff_low
    rows, cols = shape[:2]
    mask = np.zeros((rows, cols))
    mid_row, mid_col = int(rows / 2), int(cols / 2)
    for i in range(rows):
        for j in range(cols):
            d = math.sqrt((i - mid_row) ** 2 + (j - mid_col) ** 2)
            if d < cutoff_high:
                mask[i, j] = 0
            else:
                mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0))
    mask = mask.reshape((rows, cols, 1))
    return np.tile(mask, (1, 1, 3))


def transform_band_pass_filter(img, mask):
    '''
        Perform band pass filtering
    '''
    # 1. FFT
    fft = np.fft.fft2(img, axes = (0, 1))

    # 2. Shift the fft to the center of the low frequencies
    shift_fft = np.fft.fftshift(fft, axes = (0, 1))

    # 3. Filter the image frequency based on the mask
    filtered_image = np.multiply(mask, shift_fft)

    # 4. Compute the inverest shift
    shift_ifft = np.fft.ifftshift(filtered_image, axes = (0, 1))

    # 5. Compute the inverse fourier transform
    ifft = np.fft.ifft2(shift_ifft, axes = (0, 1))
    ifft = np.abs(ifft)

    return ifft.astype('uint8')


def get_train_transform(input_size=224):
    # return transforms.Compose([
    #     transforms.ToPILImage(),
    #     #transforms.RandomErasing(),
    #     transforms.Resize([input_size,input_size]),
    #     transforms.ColorJitter(0.15, 0.15, 0.15),
    #     transforms.RandomCrop(input_size, padding=6),  #从图片中随机裁剪出尺寸为 input_size 的图片，如果有 padding，那么先进行 padding，再随机裁剪 input_size 大小的图片
    #     Cutout(0.2),

    #     transforms.RandomHorizontalFlip(),
 
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         [0.485, 0.456, 0.406],
    #         [0.229, 0.2254, 0.225])
    # ])
    return A.Compose([
        A.Resize(width=input_size, height=input_size, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                                            intensity=(0.2, 0.5), p=0.2),
        A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2,
                                                            contrast_limit=0.2,
                                                            brightness_by_max=True,
                                                            always_apply=False, p=0.3),
        A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.2254, 0.225])
    ])

def get_val_transform(input_size=224):
    # return transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize([input_size,input_size]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         [0.485, 0.456, 0.406],
    #         [0.229, 0.2254, 0.225])
    # ])
    return A.Compose([
        A.Resize(width=input_size, height=input_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.2254, 0.225])
    ])

'''
test_mode: False or val/dev/test
'''
class MX_WFAS(Dataset):
    def __init__(self, path_imgrec, path_imgidx, input_size, test_mode=False, scale=1.0):
        super(MX_WFAS, self).__init__()
        self.test_mode = test_mode
        if self.test_mode:
            self.transform = get_val_transform(input_size)
        else:
            self.transform = get_train_transform(input_size)
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.imgidx = np.array(list(self.imgrec.keys))
        self.scale = scale
        self.kernel_bpf = get_gaussian_band_pass_filter((input_size, input_size))
        self.input_size = input_size

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        labels = header.label
        sample = mx.image.imdecode(img).asnumpy()  # RGB
        bbox = labels[2:6].astype(np.int32)
        label = int(labels[0])
        labels = [label, int(labels[1]) + 1]  # 0: live, 1->n: spoof_type
        
        # crop face bbox
        # scale = np.random.uniform(1.0, 1.2)
        bbox = _get_new_box(sample.shape[1], sample.shape[0], bbox, scale=self.scale)
        sample = sample[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
        sample = cv2.resize(sample, (self.input_size, self.input_size))
        sample_filted = transform_band_pass_filter(sample, self.kernel_bpf.copy())
        # cv2.imwrite("./tmp.jpg", sample)
        
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
            sample_filted = self.transform(image=sample_filted)["image"]
        sample = np.transpose(sample, (2, 0, 1)).astype(np.float32)
        sample_filted = np.transpose(sample_filted, (2, 0, 1)).astype(np.float32)


        if self.test_mode:
            return idx, sample, torch.tensor(labels, dtype=torch.long)
        return (sample, sample_filted), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.imgidx)


if __name__ == "__main__":
    train_set = MX_WFAS(
        path_imgrec="/mnt/sdc1/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/test_4.0.rec",
        path_imgidx="/mnt/sdc1/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/test_4.0.idx",
        input_size=224,
        test_mode=False,
        scale=1.0
    )

    while True:
        idx = np.random.randint(0, len(train_set))
        train_set[idx]
    print(train_set[0])
    print("Done")
