# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了Torch函数库字典形式调用。

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2021/12/23 15:34:52
"""

from .config import Config
from .fileio import load, dump
from .logger import get_root_logger
from .seed import seed_everywhere
from .dist import get_rank, get_world_size, init_distributed
from .cv_util import distance_pt


__all__ = ['Config', 'load', 'dump', 'get_root_logger', 'seed_everywhere',
           'get_rank', 'get_world_size', 'init_distributed', 'distance_pt']
