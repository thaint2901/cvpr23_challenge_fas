# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了encoders模型汇总调用，嵌入了timm模型库以及自定义模型结构。

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2021/12/06 18:22:06
"""

import torch
import timm.models as tm
from .cdcn import cdcnet, cdcnet_pp
from .mobilefacenet import MobileFaceNet
from timm.models.helpers import adapt_input_conv


local_pretrained_dict = {
    'MobileFaceNet': 'pretrained/model_mobilefacenet.pth',
}


def load_pretrained(model,
                    pretrained_dir,
                    default_cfg=None,
                    num_classes=1000,
                    in_chans=3,
                    filter_fn=None,
                    strict=True):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        pretrained_dir (str): local dir of pretrained weights
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        progress (bool): enable progress bar for weight download

    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    state_dict = torch.load(pretrained_dir, map_location='cpu')
    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = default_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                print(f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                print(f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = default_cfg.get('classifier', None)
    label_offset = default_cfg.get('label_offset', 0)
    if classifiers is not None or num_classes:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != default_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                del state_dict[classifier_name + '.weight']
                del state_dict[classifier_name + '.bias']
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)


def encoders(encoder_cfg):
    _encoder_cfg = encoder_cfg.copy()
    etype = _encoder_cfg.pop('type')
    pretrained = _encoder_cfg.pop('pretrained')
    if etype in ['cdcnet', 'cdcnet_pp', 'MobileFaceNet']:
        model = eval(etype)(**_encoder_cfg)
        if pretrained and etype in local_pretrained_dict:
            print(f"INFO: Init weight from {local_pretrained_dict[etype]}...")
            model.load_state_dict(torch.load(local_pretrained_dict[etype], map_location='cpu'))

    else:
        model = eval('tm.{}'.format(etype))(**_encoder_cfg)
    return model