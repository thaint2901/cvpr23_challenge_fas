# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了test.py。

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2021/12/30 19:06:47
"""

import os
import time
import torch
import argparse
from torch.nn.parallel import DataParallel
from utils import Config, get_root_logger
from apis import Runner, build_models, build_datasets, build_dataloaders


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--result_file', default='results.txt', help='the result file to save')
    parser.add_argument('--img_prefix', default=None, help='A folder of data')
    parser.add_argument('--gpus', type=int, default=None, help='the number of gpus to use')
    parser.add_argument('--thr', type=float, default=0.6931, help='thr of dev')
    args = parser.parse_args()

    return args


def main():
    """main"""
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.load_from is not None:
        cfg.check_cfg.load_from = args.load_from
    
    if args.img_prefix is not None:
        cfg.data.test.img_prefix = args.img_prefix

    if isinstance(args.gpus, int):
        cfg.gpu_ids = range(args.gpus)

    os.makedirs(os.path.expanduser(os.path.abspath(cfg.work_dir)), exist_ok=True)
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')
    cfg.log_cfg.filename = log_file

    model = build_models(cfg.model)
    model = DataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    dataset = build_datasets(cfg.data.test)
    dataloader = build_dataloaders(cfg.data.test_loader, dataset)
    # logger.info(f'Test dataset: {dataset.groups}')

    runner = Runner(
        model,
        logger,
        work_dir=cfg.work_dir,
        log_cfg=cfg.log_cfg,
        eval_cfg=cfg.eval_cfg,
        check_cfg=cfg.check_cfg)
    runner.test(dataloader, resfile=os.path.basename(cfg.work_dir)+f'res_{dataloader.dataset.test_mode}.txt', thr=args.thr)


if __name__ == '__main__':
    main()
