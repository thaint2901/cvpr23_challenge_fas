work_dir = 'output_tmp/fas_mbnv3_16_224_dropout'
gpu_ids = range(0, 1)
model = dict(
    type='DANN',
    encoder=dict(
        type='mobilenetv3_small_100',
        pretrained=True),
    cls_cfg=dict(drop_rate=0.2),
    adv_cfg=dict(max_iter=10000, output_dim=10),  # num attrs
    feat_dim=1024,
    test_cfg=dict(return_label=True, return_feature=False),
    train_cfg=dict(w_adv=1.0, w_bce=1.0))
data = dict(
    train=dict(
        type='MX_WFAS',
        path_imgrec='/mnt/sdc1/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train_4.0.rec',
        path_imgidx='/mnt/sdc1/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train_4.0.idx',
        input_size=224,
        test_mode=False,
        ram_cache=False,
        scale=2.7),
    val=dict(
        type='MX_WFAS',
        path_imgrec='/mnt/nvme0n1p2/datasets/face/dyno/spoofing_test/spoofing_test_4.0.rec',
        path_imgidx='/mnt/nvme0n1p2/datasets/face/dyno/spoofing_test/spoofing_test_4.0.idx',
        input_size=224,
        test_mode="val",
        scale=2.7),
    test=dict(
        type='MX_WFAS',
        path_imgrec='/mnt/nvme0n1p2/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/dev_4.0.rec',
        path_imgidx='/mnt/nvme0n1p2/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/dev_4.0.idx',
        input_size=224,
        test_mode="dev",
        scale=2.7),
    train_loader=dict(
        num_gpus=len(gpu_ids), shuffle=True, samples_per_gpu=16, workers_per_gpu=4),
    test_loader=dict(
        num_gpus=len(gpu_ids),
        shuffle=False,
        samples_per_gpu=32,
        workers_per_gpu=4))  ## batch_size
log_cfg = dict(
    interval=20,
    filename=None,
    plog_cfg=None)
eval_cfg = dict(interval=40, score_type='acer')
optim_cfg = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
sched_cfg = dict(type='CosineLR', total_epochs=15, gamma=0.01, warmup=3000)
check_cfg = dict(
    interval=15000,
    save_topk=3,
    load_from=None,
    resume_from=None,
    pretrain_from=None)
total_epochs = 40
