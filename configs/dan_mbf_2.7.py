work_dir = 'output/dan_mbf_2.7'
gpu_ids = range(0, 1)
model = dict(
    type='DANN',
    encoder=dict(
        type='MobileFaceNet',
        pretrained=True,
        embedding_size=512),
    cls_cfg=dict(drop_rate=0.2),
    adv_cfg=dict(max_iter=40000, output_dim=10),  # num attrs
    feat_dim=512,
    test_cfg=dict(return_label=True, return_feature=False),
    train_cfg=dict(w_adv=0.1, w_bce=1.0))
data = dict(
    train=dict(
        type='MX_WFAS',
        path_imgrec='/root/train_4.0.rec',
        path_imgidx='/root/train_4.0.idx',
        input_size=112,
        test_mode=False,
        ram_cache=True,
        scale=2.7),
    val=dict(
        type='MX_WFAS',
        path_imgrec='/mnt/nvme0n1p2/datasets/face/dyno/spoofing_test/spoofing_test_4.0.rec',
        path_imgidx='/mnt/nvme0n1p2/datasets/face/dyno/spoofing_test/spoofing_test_4.0.idx',
        input_size=112,
        test_mode='val',
        scale=2.7),
    test=dict(
        type='MX_WFAS',
        path_imgrec='/root/dev_4.0.rec',
        path_imgidx='/root/dev_4.0.idx',
        input_size=112,
        test_mode='dev',
        scale=2.7),
    train_loader=dict(
        num_gpus=1, shuffle=True, samples_per_gpu=256, workers_per_gpu=12),
    test_loader=dict(
        num_gpus=1, shuffle=False, samples_per_gpu=32, workers_per_gpu=4))
log_cfg = dict(interval=20, filename=None, plog_cfg=None)
eval_cfg = dict(interval=1000, score_type='acer')
optim_cfg = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
sched_cfg = dict(type='CosineLR', total_epochs=25, gamma=0.01, warmup=3000)
check_cfg = dict(
    interval=15000,
    save_topk=3,
    load_from=None,
    resume_from=None,
    pretrain_from=None)
total_epochs = 50
