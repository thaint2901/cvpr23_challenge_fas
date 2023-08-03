# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了CLS配置文件。

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2022/02/15 11:19:44
"""
work_dir = 'output/clsbce_vit_large_16_224_12_dev_40'
gpu_ids = range(0, 4)
model = dict(
    type='CLS',
    encoder=dict(
        type='vit_large_patch16_224',
        pretrained=True,
        num_classes=2,
        features_only=False),
	feat_dim=(1024,1),
    test_cfg=dict(
        return_label=True,
        return_feature=False),
    train_cfg=dict(
        w_bce=1.0))
data_root = './'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5],
    std=[255, 255, 255],
    to_rgb=False)
data = dict(
    train=dict(
        type='FasDataset',
        data_root="data",
        ann_files=['train_label.txt', 'dev_label.txt'],
        img_prefix= ['/home/cvpr23_fas_data/'],
        test_mode=False,
        pipeline=[
            dict(
                CoarseDropout=dict(
                    max_holes=1,
                    max_height=200,
                    max_width=200,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.2),
                RandomFlip=dict(hflip_ratio=0.5, vflip_ratio=0),
                RandomRotate=dict(max_angle=15, rotate_ratio=0.5),
                PatchShuffle=dict(num_patch=3, shuffer_ratio=0.3),
                RandomCrop=dict(crop_ratio=0.3, crop_range=(0.2, 0.2)),
                PhotoMetricDistortion=dict(
                    graying=0.1,
                    motion_blur=0.1,
                    blur_sharpen=0.2,
                    swap_channels=0.1),
                RandomResize=dict(ratio=0.5, scale=(32, 32)),
                Resize=dict(scale=(224, 224)),
                Normalize=dict(
                    mean=[127.5, 127.5, 127.5],
                    std=[255, 255, 255],
                    to_rgb=False))
        ]),
    val=dict(
        type='FasDataset',
        data_root="data",
        ann_files=['val_1w.txt', 'dev_label_1w.txt'],
        img_prefix= ['/home/cvpr23_fas_data/'],
        test_mode=True,
        pipeline=dict(
            Resize=dict(scale=(224, 224)),
            Normalize=dict(**img_norm_cfg))),
    test=dict(
        type='FasDataset',
        data_root="data",
        ann_files=['dev_label.txt', 'test.txt'],
        img_prefix= ['/home/cvpr23_fas_data/'],
        test_mode=True,
        pipeline=[
            dict(            
                Resize=dict(scale=(224, 224)),
                Normalize=dict(**img_norm_cfg)),
            # dict(
            #     RandomFlip=dict(hflip_ratio=1, vflip_ratio=0),
            #     Resize=dict(scale=(224, 224)),
            #     Normalize=dict(**img_norm_cfg)),
            # dict(            
            #     RandomRotate=dict(max_angle=15, rotate_ratio=1),
            #     Resize=dict(scale=(224, 224)),
            #     Normalize=dict(**img_norm_cfg)),
            # dict(            
            #     RandomCrop=dict(crop_ratio=1, crop_range=(0.2, 0.2)),
            #     Resize=dict(scale=(224, 224)),
            #     Normalize=dict(**img_norm_cfg)),  
        ]),
    train_loader=dict(
        num_gpus=len(gpu_ids),
        shuffle=True,
        samples_per_gpu=2,
        workers_per_gpu=4),
    test_loader=dict(
        num_gpus=len(gpu_ids),
        shuffle=False,
        samples_per_gpu=128,
        workers_per_gpu=4))
log_cfg = dict(
    interval=20,
    filename=None,
    plog_cfg=dict(
        loss_types='all',
        eval_types=['acer', 'apcer', 'bpcer']))
eval_cfg = dict(
    interval=200,
    score_type='acer',
    tsne_cfg=dict(
        marks=None,
        filename='_tsne.png',
        maxsamples=20000))
optim_cfg = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0)
sched_cfg = dict(type='CosineLR', total_epochs=40, gamma=0.01, warmup=3000)
check_cfg = dict(
    interval=5e4,
    save_topk=3,
    load_from=None,
    resume_from=None,
    pretrain_from=None)
total_epochs = 40
