# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件的MADDG结构, CVPR19论文活体模型 (Multi-adversarial discriminative deep domain generalization
for face presentation attack detection)

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2021/12/06 18:23:09
"""

import torch
import torch.nn as nn


class MADDG(nn.Module):
    """implementation of `Multi-adversarial discriminative deep domain generalization
    for face presentation attack detection <https://openaccess.thecvf.com/content_CVPR_2019/>

    """
    def __init__(self,
                 num_classes=2,
                 pretrained=False,
                 features_only=True):
        super(MADDG, self).__init__()
        self.features_only = features_only
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(196)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(128)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_5 = nn.BatchNorm2d(128)
        self.relu1_5 = nn.ReLU(inplace=True)
        self.conv1_6 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_6 =  nn.BatchNorm2d(196)
        self.relu1_6 = nn.ReLU(inplace=True)
        self.conv1_7 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_7 = nn.BatchNorm2d(128)
        self.relu1_7 = nn.ReLU(inplace=True)
        self.maxpool1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_8 = nn.BatchNorm2d(128)
        self.relu1_8 = nn.ReLU(inplace=True)
        self.conv1_9 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_9 = nn.BatchNorm2d(196)
        self.relu1_9 = nn.ReLU(inplace=True)
        self.conv1_10 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_10 = nn.BatchNorm2d(128)
        self.relu1_10 = nn.ReLU(inplace=True)
        self.maxpool1_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_layer_1 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_1,
            self.conv3_2,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.pool2_2,
            self.conv3_3,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward_features(self, img):
        """features forward"""
        out = self.conv1_1(img)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv1_3(out)
        out = self.bn1_3(out)
        out = self.relu1_3(out)
        out = self.conv1_4(out)
        out = self.bn1_4(out)
        out = self.relu1_4(out)
        pool_out1 = self.maxpool1_1(out)

        out = self.conv1_5(pool_out1)
        out = self.bn1_5(out)
        out = self.relu1_5(out)
        out = self.conv1_6(out)
        out = self.bn1_6(out)
        out = self.relu1_6(out)
        out = self.conv1_7(out)
        out = self.bn1_7(out)
        out = self.relu1_7(out)
        pool_out2 = self.maxpool1_2(out)

        out = self.conv1_8(pool_out2)
        out = self.bn1_8(out)
        out = self.relu1_8(out)
        out = self.conv1_9(out)
        out = self.bn1_9(out)
        out = self.relu1_9(out)
        out = self.conv1_10(out)
        out = self.bn1_10(out)
        out = self.relu1_10(out)
        pool_out3 = self.maxpool1_3(out)

        feature = self.bottleneck_layer_1(pool_out3)
        feature = self.avg_pool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)

        return feature

    def forward(self, img):
        """forward"""
        out = self.forward_features(img)
        if self.features_only:
            return out
        out = self.fc(out)

        return out


def maddg(pretrained=False, **kwargs):
    """Constructs a MADDG model.
    """
    return MADDG(**kwargs)
