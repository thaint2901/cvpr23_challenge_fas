FAS
===
### 特点
几乎所有的网络配置，超参数都能在配置文件中进行修改，甚至网络结构。
### Backbone
* Timm库，CDCN, CDCN++
### 活体检测方法
* CLS, DAN;
* 训练策略：分布式训练，渐进式训练

依赖库
opencv-python          4.6.0.66
timm                   0.5.4
tqdm                   4.64.1
torch                  1.11.0+cu113

快速开始
---
推理
python test.py 'configs/dan_vitl_16_224_dropout.py' --load_from 'configs/top1_model.pth' --img_prefix '/path/to/data' --gpu 1

训练 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train.py 'configs/dan_vitl_16_224_dropout.py' > 0train.log 2>&1 &
---

### owners
* zouzhaofan(zouzhf41@chinatelecom.cn)
* xuyaowen(xuyw1@chinatelecom.cn)
Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
