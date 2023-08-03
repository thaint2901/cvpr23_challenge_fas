# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了CDCN(Searching Central Difference Convolutional
    Networks for Face Anti-Spoofing) OP: https://arxiv.org/pdf/2003.04092v1.pdf

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2022/01/30 12:21:31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ConvNormAct


class CDCNet(nn.Module):
    """implementation of `Searching Central Difference Convolutional
    Networks for Face Anti-Spoofing <https://arxiv.org/pdf/2003.04092v1.pdf>

    """
    def __init__(self,
                 theta=0.7,
                 num_classes=2,
                 pretrained=True,
                 features_only=False,
                 conv_cfg=dict(type='CDCConv2d'),
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='ReLU')):
        super(CDCNet, self).__init__()
        conv_cfg['theta'] = theta
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.features_only = features_only
        self.conv1 = ConvNormAct(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.block1 = self._make_block([64, 128, 196, 128])
        self.block2 = self._make_block([128, 128, 196, 128])
        self.block3 = self._make_block([128, 128, 196, 128])
        self.lateral1 = ConvNormAct(
            3 * 128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.lateral2 = ConvNormAct(
            128,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.features_only:
            self.lateral3 = ConvNormAct(
                64,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=act_cfg)
        else:
            self.fc = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)

    def _make_block(self, channels):
        """make block"""
        layers = list()
        for i in range(1, len(channels)):
            layers.append(
                ConvNormAct(
                    channels[i - 1],
                    channels[i], 3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward_features(self, img):
        """features forward"""
        out = self.conv1(img)
        out1 = self.block1(out)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out_cat = [F.interpolate(x, (32, 32), mode='bilinear', align_corners=False) for x in [out1, out2, out3]]
        out_cat = torch.cat(out_cat, dim=1)
        out = self.lateral1(out_cat)
        out = self.lateral2(out)
        if self.features_only:
            out = self.lateral3(out)
        else:
            out = F.adaptive_avg_pool2d(out, 1)
            out = self.fc(out)
        return [out1, out2, out3, out_cat, out]

    def forward(self, img):
        """forward"""
        outs = self.forward_features(img)
        if self.features_only:
            return outs
        else:
            return outs[-1].squeeze(3).squeeze(2)


class CDCNetpp(nn.Module):
    """implementation of `Searching Central Difference Convolutional
    Networks for Face Anti-Spoofing <https://arxiv.org/pdf/2003.04092v1.pdf>

    """
    def __init__(self,
                 theta=0.7,
                 num_classes=2,
                 pretrained=True,
                 features_only=False,
                 conv_cfg=dict(type='CDCConv2d'),
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='ReLU')):
        super(CDCNetpp, self).__init__()
        conv_cfg['theta'] = theta
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.features_only = features_only
        self.conv1 = ConvNormAct(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.block1 = self._make_block([64, 128, 204, 128])
        self.block2 = self._make_block([128, 153, 128, 179, 128])
        self.block3 = self._make_block([128, 128, 153, 128])

        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)

        if self.features_only:
            self.lateral1 = nn.Sequential(
                ConvNormAct(
                    3 * 128,
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvNormAct(
                    128,
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=act_cfg))
        else:
            self.lateral1 = ConvNormAct(
                3 * 128,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fc = nn.Conv2d(128, num_classes, kernel_size=1, bias=True)

    def _make_block(self, channels):
        """make block"""
        layers = list()
        for i in range(1, len(channels)):
            layers.append(
                ConvNormAct(
                    channels[i - 1],
                    channels[i], 3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward_features(self, img):
        """features forward"""
        out = self.conv1(img)
        out1 = self.block1(out)
        out1_c = F.interpolate(self.sa1(out1) * out1, (32, 32), mode='bilinear', align_corners=False)

        out2 = self.block2(out1)
        out2_c = F.interpolate(self.sa2(out2) * out2, (32, 32), mode='bilinear', align_corners=False)

        out3 = self.block3(out2)
        out3_c = F.interpolate(self.sa3(out3) * out3, (32, 32), mode='bilinear', align_corners=False)

        out_cat = torch.cat([out1_c, out2_c, out3_c], dim=1)
        out = self.lateral1(out_cat)
        if not self.features_only:
            out = F.adaptive_avg_pool2d(out, 1)
            out = self.fc(out)
        return [out1, out2, out3, out_cat, out]

    def forward(self, img):
        """forward"""
        outs = self.forward_features(img)
        if self.features_only:
            return outs
        else:
            return outs[-1].squeeze(3).squeeze(2)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """forward"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


def cdcnet(pretrained=False, **kwargs):
    """Constructs a CDCNet model.
    """
    return CDCNet(**kwargs)


def cdcnet_pp(pretrained=False, **kwargs):
    """Constructs a CDCNetpp model.
    """
    return CDCNetpp(**kwargs)