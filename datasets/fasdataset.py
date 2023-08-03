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

import os
import cv2
import copy
import warnings
import random
import numpy as np
from utils.fileio import load
from utils.cv_util import distance_pt
from datasets.transform import Transforms

class FasDataset():
    """YeWu dataset for FAS.
    The annotation format is show as follows:
        -- annotation.txt
            ...
            img_path/img_name.jpg x_1 y_1 ... x_72 y_72 label
            ...
    Args:
        ann_file (str or list[str]): Annotation file path
        pipeline (dict): Processing pipeline.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``mask_prefix``,  if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        enlarge (float): enlarge face to crop according to landmarks.
    """
    NAME = "FasDataset"

    def __init__(self,
                 data_root,
                 ann_files,
                 pipeline=None,
                 img_prefix='',
                 test_mode=False,
                 enlarge=3.0,
                 ):
        self.ann_files = ann_files if isinstance(ann_files, list) else [ann_files]
        self.img_prefix = img_prefix if isinstance(img_prefix, list) else [img_prefix]
        self.data_root = data_root
        self.test_mode = test_mode

        if len(self.img_prefix) == 1:
            self.img_prefix *= len(self.ann_files)
        elif len(self.img_prefix) != len(self.ann_files):
            raise ValueError("num of img_prefix not equal to ann_files!")

        self.pipeline = 0
        if isinstance(pipeline, dict):
            self.pipeline_list = [Transforms(pipeline)]
        elif isinstance(pipeline, list):
            self.pipeline_list = [Transforms(p) for p in pipeline]
        else:
            raise ValueError("improper format pipeline")
        self.pipeline_num = len(self.pipeline_list)

        self.load_annotations(ann_files)
        self.groups = self.set_group_flag()

    def load_annotations(self, ann_files):
        """load annotation information"""
        ann_nums = dict()
        _labels = list()
        _filenames = list()
        _domain = list()
        for i, ann_file in enumerate(ann_files):
            with open(os.path.join(self.data_root, ann_file), 'r') as f:
                lines = f.readlines()
            ann_nums[os.path.splitext(os.path.basename(ann_file))[0]] = len(lines)
            for j, l in enumerate(lines):
                l = l.strip().split()
                try:
                    label = (1-int(l[1]))
                    # domain = 0
                except:
                    label = 0 if self.test_mode else 10
                    # domain = 1
                domain = i
                _labels.append(label)
                _filenames.append(os.path.join(self.img_prefix[i], l[0]))
                _domain.append(domain)
 
        self.labels = np.array(_labels)
        self.filenames = np.array(_filenames)
        self.domain = np.array(_domain)
        self.ann_nums = ann_nums

    def set_group_flag(self):
        """Set flag according to label"""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = self.labels[i]
        return np.bincount(self.flag)

    def _rand_another(self, idx):
        """random select another index"""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _get_ann_info(self, filename, label, domain):
        """read images and annotations"""
        img = cv2.imread(filename)
        if img is None:
            print(filename)
        data = dict(
            img=img,
            label=np.ones((1,)).astype(np.int64) * label,
            path=os.path.relpath(filename, self.img_prefix[0]) if len(set(self.img_prefix))==1 else filename,
            domain=domain
            )
        return data

    def __getitem__(self, idx):
        while True:
            label = copy.deepcopy(self.labels[idx])
            filename = copy.deepcopy(self.filenames[idx])
            domain = copy.deepcopy(self.domain[idx])
            try:
                data = self._get_ann_info(filename, label, domain)
            except:
                warnings.warn('Fail to read image: {}'.format(filename))
                idx = self._rand_another(idx)
                continue
            break
        if self.test_mode:
            data = self.pipeline_list[self.pipeline](data)
        else:
            data = self.pipeline_list[0](data)
        return data
    
    def __len__(self):
        return len(self.labels)