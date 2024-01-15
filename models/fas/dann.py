# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现域泛化DG的域对抗方法

Authors: xuyaowen(xuyw1@chinatelecom.cn)
Date:    2022/12/31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbones import encoders


class GRL(torch.autograd.Function):
    '''
    Ganin Y, Lempitsky V. Unsupervised Domain Adaptation by Backpropagation[C]//
    International Conference on Machine Learning. 2015: 1180-1189.
    '''
    gamma = 10
    p = 0
    @staticmethod
    def forward(ctx, input):
        ctx.constant = 2.0 / (1.0 + np.exp(-GRL.gamma * GRL.p)) - 1.0 # coeff
        return input * 1.0
    @staticmethod
    def backward(ctx, gradOutput):
        return -ctx.constant * gradOutput


class Discriminator(nn.Module):
    '''
    cvpr2020论文(Single-Side Domain Generalization for Face Anti-Spoofing)
    '''
    def __init__(self, max_iter, input_dim=2048, output_dim=10):
        super(Discriminator, self).__init__()
        self.max_iter = max_iter
        self.iter_num = 0.
        self.gamma = 10
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, output_dim)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )

    def forward(self, feature):
        self.iter_num += 1
        GRL.p = self.iter_num / self.max_iter
        adversarial_out = self.ad_net(GRL.apply(feature))
        return adversarial_out


class Classifier(nn.Module):
    def __init__(self, feat_dim, drop_rate):
        super(Classifier, self).__init__()
        self.drop_rate = drop_rate
        self.classifier_layer = nn.Linear(feat_dim, 1)  # bce
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.dropout(x, p=self.drop_rate, training=self.training)

        classifier_out = self.classifier_layer(x)
        return classifier_out


class DANN(nn.Module):
    """Domain adversarial learning for face anti-spoofing.

    Args:
        encoder (dict): the config dict of encoder network.
        adv_cfg (dict): the config dict of adv layer.
        feat_dim (tuple or None): in_channels and out_channels of extra conv,
            (in_channels, out_channels).
        test_cfg (dict): the config dict of testing setting.
        train_cfg (dict): the config dict of training setting, including
            some hyperparameters of loss.
    """
    def __init__(self,
                 encoder,
                 cls_cfg,
                 adv_cfg,
                 feat_dim=1024,
                 test_cfg=None,
                 train_cfg=None):
        super(DANN, self).__init__()
        assert isinstance(encoder, dict)
        assert isinstance(cls_cfg, dict)
        assert isinstance(adv_cfg, dict)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.return_label = self.test_cfg.pop('return_label', True)
        self.return_feature = self.test_cfg.pop('return_feature', False)

        self.encoder = encoders(encoder)
        self.adv = Discriminator(**adv_cfg, input_dim=feat_dim)
        self.classifer = Classifier(**cls_cfg, feat_dim=feat_dim)

        self.cls_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _get_losses(self, feats, label):
        """calculate training losses"""
        loss_bce = self.bce_loss(feats[0], (1-label[:, :1]).to(torch.float)) * self.train_cfg['w_bce']
        loss_adv = self.cls_loss(feats[1], label[:, 1]) * self.train_cfg['w_adv']
        loss = loss_bce + loss_adv
        return dict(loss_bce=loss_bce, loss_adv=loss_adv, loss=loss)

    def forward(self, img, label=None):
        """forward"""
        feat = self.encoder.forward_features(img)
        feat = self.encoder.forward_head(feat, pre_logits=True)

        out = self.classifer(feat)
        if self.training:
            ad_out = self.adv(feat)
            losses = self._get_losses([out, ad_out], label)
            return losses
        else:
            pred = torch.sigmoid(out/5)[:, 0]
            output = [pred]
            if self.return_label:
                output.append(label)
            if self.return_feature:
                output.append(feat)
            return output
