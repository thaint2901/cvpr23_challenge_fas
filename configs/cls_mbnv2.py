work_dir = 'output/cls_mbnv2'
gpu_ids = range(0, 1)
model = dict(
    type='Classifier',
    encoder=dict(
        type='mobilenetv2_100',
        pretrained=True,
        num_classes=2,
        drop_rate=0.2,
        features_only=False),
    test_cfg=dict(return_label=True, return_feature=False),
    train_cfg=dict(w_cls=1.0))
data = dict(
    train=dict(
        type='MXMulti_FAS',
        data_paths=[
            '/root/train_4.0', '/root/zalo_liveness/zalo_liveness_small'
        ],
        input_size=224,
        test_mode=False,
        ram_cache=True,
        scale=2.7),
    val=dict(
        type='MXMulti_FAS',
        data_paths=['/mnt/nvme0n1p2/datasets/face/dyno/spoofing_test/spoofing_test_4.0'],
        input_size=224,
        test_mode='val',
        scale=2.7),
    test=dict(
        type='MXMulti_FAS',
        data_paths=[
            '/mnt/nvme0n1p2/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/dev_4.0'
        ],
        input_size=224,
        test_mode='dev',
        scale=2.7),
    train_loader=dict(
        num_gpus=len(gpu_ids), shuffle=True, samples_per_gpu=256, workers_per_gpu=16),
    test_loader=dict(
        num_gpus=len(gpu_ids),
        shuffle=False,
        samples_per_gpu=32,
        workers_per_gpu=4))  ## batch_size
log_cfg = dict(interval=20, filename=None, plog_cfg=None)
eval_cfg = dict(interval=400, score_type='acer')
optim_cfg = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
sched_cfg = dict(type='CosineLR', total_epochs=20, gamma=0.01, warmup=3000)
check_cfg = dict(
    interval=15000,
    save_topk=3,
    load_from="latest.pth",
    resume_from=None,
    pretrain_from=None)
total_epochs = 40
