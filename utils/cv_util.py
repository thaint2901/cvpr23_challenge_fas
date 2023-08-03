# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了cv相关的一些图像处理函数
Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2022/09/29 16:48:57
"""

import cv2

def distance_pt(pt1, pt2):
    return ((pt2[1] - pt1[1])**2 +  (pt2[0] - pt1[0])**2)**0.5