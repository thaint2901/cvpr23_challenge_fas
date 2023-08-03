# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现 单中心 分类方法

Authors: xuyaowen(xuyw1@chinatelecom.cn)
Date:    2022/12/27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import encoders
from models.losses import OneCenterLoss

class CLS(nn.Module):
    """General classification network.

    Args:
        Args:
        encoder (dict): the config dict of encoder network.
        feat_dim (tuple): in_channels and out_channels of fc layer,
            (in_channels, out_channels).
        test_cfg (dict): the config dict of testing setting.
        train_cfg (dict): the config dict of training setting, including
            some hyperparameters of loss.

    """
    def __init__(self,
                 encoder,
                 feat_dim=(2048, 1),
                 test_cfg=None,
                 train_cfg=None):
        super(CLS, self).__init__()
        assert isinstance(encoder, dict)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.return_label = self.test_cfg.pop('return_label', True)
        self.return_feature = self.test_cfg.pop('return_feature', False)

        self.encoder = encoders(encoder)
        self.fc = nn.Linear(feat_dim[0], feat_dim[1], bias=True)
        self.features_only = encoder.pop('features_only', False)

        self.oc_loss = OneCenterLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _get_losses(self, feats, label):
        """calculate training losses"""
        if 'w_bce' in self.train_cfg and 'w_oc' in self.train_cfg:
            loss_bce = self.bce_loss(feats[0][:, 0], 1 - label[:, 0].float()).unsqueeze(0) * self.train_cfg['w_bce']
            loss_oc = self.oc_loss(feats[1], label[:, 0]).unsqueeze(0) * self.train_cfg['w_oc']
            loss = loss_bce + loss_oc
            return dict(loss_bce=loss_bce, loss_oc=loss_oc, loss=loss)
        if 'w_oc' in self.train_cfg:
            loss_oc = self.oc_loss(feats[1], label[:, 0]).unsqueeze(0) * self.train_cfg['w_oc']
            return dict(loss_oc=loss_oc, loss=loss_oc)
        if 'w_bce' in self.train_cfg:
            loss_bce = self.bce_loss(feats[0][:, 0], 1 - label[:, 0].float()).unsqueeze(0) * self.train_cfg['w_bce']
            return dict(loss_bce=loss_bce, loss=loss_bce)

    def forward(self, img, label=None, domain=None):
        """forward"""
        if self.features_only:
            feat = self.encoder(img)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(3).squeeze(2)
        else:
            feat = self.encoder.forward_features(img)
        out = self.fc(feat)
        if self.training:
            losses = self._get_losses([out, feat], label)
            return losses
        else:
            pred = torch.sigmoid(out/5)[:, 0]
            output = [pred]
            if self.return_label:
                output.append(label)
            if self.return_feature:
                output.append(feat)
            return output

