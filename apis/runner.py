# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现运行器 Runner.

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2021/12/24 15:29:40
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from apis.evaluator import Metric
from utils.dist import is_main_process
from apis.visualizer import VisualizeLog, VisualizeTSNE
from apis.builder import build_optimizers, build_schedulers, build_dataloaders

# def load_model(load_path, model, optimizer = None, allow_size_mismatch = False):
#     """
#     Load model from file
#     If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
#     If optimizer not passed, only the model weights will be loaded
#     """

#     # -- load dictionary
#     assert os.path.isfile( load_path ), "Error when loading the model, provided path not found: {}".format( load_path )
#     checkpoint = torch.load(load_path)
#     loaded_state_dict = checkpoint['state_dict']

#     if allow_size_mismatch:
#         loaded_sizes = { k: v.shape for k,v in loaded_state_dict.items() }
#         model_state_dict = model.state_dict()
#         model_sizes = { k: v.shape for k,v in model_state_dict.items() }
#         mismatched_params = []
#         for k in loaded_sizes:
#             if loaded_sizes[k] != model_sizes[k]:
#                 mismatched_params.append(k)
#         for k in mismatched_params:
#             del loaded_state_dict[k]

#     # -- copy loaded state into current model and, optionally, optimizer
#     model.load_state_dict(loaded_state_dict, strict = not allow_size_mismatch)
#     if optimizer is not None:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         return model, optimizer, checkpoint['epoch_idx'], checkpoint
#     return model

class Runner(object):
    """training or testing system.

    Args:
        model (nn.Module): the model to train/test.
        logger (logger): the logger for train/test.
        work_dir (str): work dir to save checkpoint or log file.
        eval_cfg (dict): the config dict of evaluation.
        optim_cfg (dict): the config dict of optimizer.
        sched_cfg (dict): the config dict of learning rate schedule.
        check_cfg (dict): the config dict of checkpoint.
    """
    def __init__(self,
                 model,
                 logger,
                 work_dir,
                 log_cfg=None,
                 eval_cfg=None,
                 optim_cfg=None,
                 sched_cfg=None,
                 check_cfg=None):
        assert isinstance(work_dir, str)
        self.model = model
        self.logger = logger
        self.log_cfg = log_cfg
        self.eval_cfg = eval_cfg
        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.check_cfg = check_cfg
        self.work_dir = os.path.abspath(work_dir)
        self._total_epoch = 0
        self._total_iter = 0
        self._iter_time = 0
        self._epoch = 0
        self._iter = 0
        self._eta = 0
        self._lr = 0

        if self.optim_cfg is not None:
            self._warmup = self.sched_cfg.pop('warmup', 0)
            self.optimizer = build_optimizers(self.optim_cfg, self.model)
            self.scheduler = build_schedulers(self.sched_cfg, self.optimizer)
            # if self.log_cfg.plog_cfg is not None:
            #     self.vis_log = VisualizeLog(self.work_dir, self.log_cfg.plog_cfg)
            #     self.writer_log = LogWriter(logdir=self.work_dir)
            #     hparm_cfg = self.log_cfg.plog_cfg.pop('hparm_cfg', None)
            #     if hparm_cfg is not None:
            #         self.writer_log.add_hparams(
            #             hparams_dict=dict(hparm_cfg),
            #             metrics_list=self.log_cfg.plog_cfg.eval_types + self.log_cfg.plog_cfg.loss_types)

        self._score = np.zeros((self.check_cfg.pop('save_topk', 1),), dtype=np.float32)
        self._init_model(self.check_cfg.resume_from, self.check_cfg.load_from, 
                         self.check_cfg.pretrain_from)
        self.metric = Metric(logger, self.work_dir, eval_cfg)

        # if self.eval_cfg.tsne_cfg is not None:
        #     self.vis_tsne = VisualizeTSNE(self.work_dir, self.log_cfg.filename, self.eval_cfg.tsne_cfg.copy())

        self.dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def _lr_step(self):
        """update learning rate"""
        if self._warmup > self._iter:
            init_lrs = [v['initial_lr'] for v in self.optimizer.param_groups]
            for param_group, lr in zip(self.optimizer.param_groups, init_lrs):
                param_group['lr'] = lr / self._warmup * self._iter
        self._lr = self.optimizer.param_groups[0]['lr']

    def _log_infos(self, output):
        """log training info"""
        eta = (self._total_iter - self._iter) * self._iter_time / self.log_cfg.interval
        mins = '{:2d}'.format(int((eta % 3600) / 60))
        hours = '{:2d}'.format(int(eta / 3600))
        lr = '{:6f}'.format(self._lr)
        self._iter_time = 0
        info = f'Epoch: {self._epoch}, Iter: {self._iter}, ETA: {hours}h{mins}min, Lr: {lr},'
        # self.writer_log.add_scalar(tag='lr', step=self._iter, value=self._lr)
        #TODO: Tensorboard instead writer_log
        for k, v in output.items():
            # if self.log_cfg.plog_cfg is not None and (k in self.log_cfg.plog_cfg.loss_types or self.log_cfg.plog_cfg.loss_types == 'all' ):
            #     self.writer_log.add_scalar(tag=k, step=self._iter, value=v.mean().detach().item())
            if k == 'loss':
                continue
            loss = '{:.5f}'.format(v.mean().detach().item())
            info += f' {k}: {loss},'
        info += ' loss: {:.5f}'.format(output['loss'].mean().detach().item())
        self.logger.info(info)

    def _init_model(self, resume_from=None, load_from=None, pretrain_from=None):
        """initialize model"""
        if resume_from is not None:
            try:
                checkpoint = torch.load(resume_from)
            except:
                checkpoint = torch.load(resume_from, map_location='cpu')
            self._iter = checkpoint['iter']
            self._epoch = checkpoint['epoch']
            self._score = checkpoint['score']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f'Resume from {resume_from}, {self._epoch} epoch, {self._iter} iter')
        elif load_from is not None:
            try:
                checkpoint = torch.load(load_from)
            except:
                checkpoint = torch.load(load_from, map_location='cpu')
            self._iter = checkpoint['iter']
            self._epoch = checkpoint['epoch']
            self._score = checkpoint['score']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f'Load from {load_from}, {self._epoch} epoch, {self._iter} iter')
        elif pretrain_from is not None:
            try:
                checkpoint = torch.load(pretrain_from)
            except:
                checkpoint = torch.load(pretrain_from, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f'Pretrain from {pretrain_from}, {self._epoch} epoch, {self._iter} iter')


    def _save_model(self, score=None, filename=None):
        """save model"""
        if score is not None:
            if score < self._score[-1]:
                return
            for k in range(len(self._score)):
                if score >= self._score[k]:
                    filename = os.path.join(self.work_dir, f'top{k + 1}_model.pth')
                    if os.path.exists(filename) and k < len(self._score) - 1:
                        self._score[k + 1] = self._score[k]
                        filename_next = os.path.join(self.work_dir, f'top{k + 2}_model.pth')
                        os.system(f'mv {filename} {filename_next}')
                    self._score[k] = score
                    break
        elif filename is None:
            filename = os.path.join(self.work_dir, f'epoch{self._epoch}_iter{self._iter}.pth')
        else:
            filename = os.path.join(self.work_dir, filename)
        checkpoint = dict(
            iter=self._iter,
            epoch=self._epoch,
            score=self._score,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict())
        torch.save(checkpoint, filename)
        self.logger.info(f'save model: {filename}')

    def _get_total_iter(self, dataloader, step_cfg=None):
        """get total iter"""
        if step_cfg is None:
            return self._total_epoch * len(dataloader)
        else:
            ir, hr, dr = step_cfg.init_rate, step_cfg.hem_rate, step_cfg.decay_rate
            train_len = int(ir * len(dataloader))
            test_len = len(dataloader) - train_len
            total_iter = 0
            if self._total_epoch < step_cfg.interval:
                self._total_iter = self._total_epoch * len(dataloader)
            for i in range(int(self._total_epoch / step_cfg.interval)):
                total_iter += train_len * step_cfg.interval
                train_len += test_len * hr
                test_len *= (1 - hr)
                hr *= dr
        self._total_iter = int(total_iter)
        return int(total_iter)

    def _train_epoch(self, dataloader):
        """train one epoch"""
        start_time = time.time()
        for i, data in enumerate(dataloader):
            self._iter += 1
            
            output = self.model(img=data[0], label=data[1])
            self.optimizer.zero_grad()
            output['loss'].mean().backward()

            self.optimizer.step()
            self._lr_step()

            self._iter_time += time.time() - start_time
            start_time = time.time()

            if self._iter % self.log_cfg.interval == 0 and is_main_process():
                self._log_infos(output)

            if self._iter % self.eval_cfg.interval == 0 and is_main_process():
                eval_cfg = self.eval_cfg.copy()
                score = self.val()
                self._save_model(score)
                self._save_model(filename=None if eval_cfg.pop('save_model', False) else 'latest.pth')

            if self._iter % self.check_cfg.interval == 0 and is_main_process():
                self._save_model()

    @torch.no_grad()
    def val(self):
        """val method"""
        # feats = list()
        preds = list()
        labels = list()
        self.model.eval()
        for data in tqdm(self.val_dataloader):
            output = self.model(img=data[0], label=data[1])
            preds.append(output[0].detach().cpu().numpy())
            labels.append(output[1].detach().cpu().numpy()[:, 0])
            # if len(output) > 2:
            #     feats.append(output[2].detach().cpu().numpy())
        self.model.train()
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        score, eval_dict = self.metric(preds, labels)
        # if len(feats) > 0:
            # feats = np.concatenate(feats)
            # self.writer_log.add_embeddings(tag='feature', mat=feats, metadata=labels.astype(str))
        # if self.log_cfg.plog_cfg is not None:
        #     for k, v in eval_dict.items():
        #         if k in self.log_cfg.plog_cfg.eval_types:
        #             self.writer_log.add_scalar(tag=k, step=self._iter, value=v)
        return score

    @torch.no_grad()
    def test(self, dataloader, resfile=None, ann_nums=None, log_info=True, thr=None):
        """test method"""
        self.model.eval()
        preds_all = list()
        for n in range(dataloader.dataset.pipeline_num):
            dataloader.dataset.pipeline = n
            self.logger.info(f'Test-Time Augmentation ({n+1}/{dataloader.dataset.pipeline_num})')
            feats = list()
            preds = list()
            paths = list()
            labels = torch.tensor([])
            preds = torch.tensor([])
            for data in tqdm(dataloader):
                paths.extend(data.pop('path', 'unknow'))
                output = self.model(**data)
                preds = torch.cat([preds, output[0].detach().cpu()])
                labels = torch.cat([labels, output[1].detach().cpu()[:, 0]])
                if len(output) > 2 and n==dataloader.dataset.pipeline_num-1:
                    feats.append(output[2].detach().cpu().numpy())
            preds_all.append(preds.unsqueeze(1))
            if dataloader.dataset.pipeline_num > 1:
                score, _ = self.metric(preds.numpy(), labels.numpy(), thr=thr, dataname=str(n+1), log_info=log_info)
        preds_all = torch.cat(preds_all, dim=1)
        preds_all = preds_all.mean(dim=1).numpy()
        labels = labels.numpy()

        score, _ = self.metric(preds_all, labels, paths=np.array(paths), thr=thr, resfile=resfile, log_info=log_info)
        if isinstance(ann_nums, dict) and len(ann_nums) >= 2:
            ind = 0
            for ann, num in ann_nums.items():
                self.metric(preds_all[ind:ind+num], labels[ind:ind+num], thr=thr, dataname=ann, log_info=log_info)
                ind += num
        if len(feats) > 0:
            feats = np.concatenate(feats)
            if self.eval_cfg.tsne_cfg is not None:
                self.logger.info('vis tsne...')
                self.vis_tsne(feats, labels, np.array(paths))
        self.logger.info('End of testing!')
        return preds_all, score

    def train(self, dataloader, cfg):
        """train method"""
        self._total_epoch = cfg.total_epochs
        self._total_iter = self._get_total_iter(dataloader)
        self.logger.info(f'Start training from the {self._epoch} Epoch, {self._iter} Iter.')
        self.logger.info(f'Total {self._total_epoch} Epochs, {self._total_iter} Iters, {len(dataloader)} Iter/Epoch.')

        self.model.train()
        start_epoch = self._epoch
        for epoch in range(start_epoch, self._total_epoch):

            self._train_epoch(dataloader)

            # if self.log_cfg.plog_cfg is not None:
            #     self.vis_log(self.log_cfg.filename)

            self._epoch += 1
            self.scheduler.step()
            self._save_model(filename='latest.pth')

        self.logger.info('End of training!')

    # def train_step(self, dataloader, cfg):
    #     """train step method"""
    #     self._total_epoch = cfg.total_epochs
    #     self._total_iter = self._get_total_iter(dataloader, cfg.step_cfg)
    #     self.logger.info(f'Start training from the {self._epoch} Epoch, {self._iter} Iter.')
    #     self.logger.info(f'Total {self._total_epoch} Epochs, {self._total_iter} Iters, {len(dataloader)} Iter/Epoch.')

    #     train_dataset = dataloader.dataset
    #     test_dataset = self.test_dataloader.dataset

    #     pselc, nselc = np.where(train_dataset.flag == 0)[0], np.where(train_dataset.flag != 0)[0]

    #     np.random.shuffle(pselc)
    #     np.random.shuffle(nselc)

    #     plmks, nlmks = train_dataset.lmks[pselc].copy(), train_dataset.lmks[nselc].copy()
    #     plabels, nlabels = train_dataset.labels[pselc].copy(), train_dataset.labels[nselc].copy()
    #     pfilenames, nfilenames = train_dataset.filenames[pselc].copy(), train_dataset.filenames[nselc].copy()

    #     ir, hr, dr = cfg.step_cfg.init_rate, cfg.step_cfg.hem_rate, cfg.step_cfg.decay_rate

    #     train_dataset.lmks = np.concatenate([plmks[:int(ir * len(plmks))].copy(), 
    #                                         nlmks[:int(ir * len(nlmks))].copy()], axis=0)
    #     train_dataset.labels = np.concatenate([plabels[:int(ir * len(plabels))].copy(), 
    #                                         nlabels[:int(ir * len(nlabels))].copy()], axis=0)
    #     train_dataset.filenames = np.concatenate([pfilenames[:int(ir * len(pfilenames))].copy(), 
    #                                         nfilenames[:int(ir * len(nfilenames))].copy()], axis=0)

    #     plmks, nlmks =  plmks[int(ir * len(plmks)):].copy(), nlmks[int(ir * len(nlmks)):].copy()
    #     plabels, nlabels = plabels[int(ir * len(plabels)):].copy(), nlabels[int(ir * len(nlabels)):].copy()
    #     pfilenames, nfilenames = pfilenames[int(ir * len(pfilenames)):].copy(), nfilenames[int(ir * len(nfilenames)):].copy()

    #     # train_dataset.data_infos = pinfos[:int(ir * len(pinfos))].tolist() + ninfos[:int(ir * len(ninfos))].tolist()
    #     # pinfos, ninfos = pinfos[int(ir * len(pinfos)):], ninfos[int(ir * len(ninfos)):]

    #     self.model.train()
    #     start_epoch = self._epoch
    #     self.logger.info(f'Train sub dataset: {train_dataset.set_group_flag()}')
    #     dataloader = build_dataloaders(cfg.data.train_loader, train_dataset)
    #     for epoch in range(start_epoch, self._total_epoch):

    #         self._train_epoch(dataloader)

    #         self._epoch += 1
    #         if self._epoch % cfg.step_cfg.interval == 0:
    #             # test_dataset.data_infos = pinfos.tolist() + ninfos.tolist()
    #             test_dataset.lmks = np.concatenate([plmks.copy(), nlmks.copy()], axis=0)
    #             test_dataset.labels = np.concatenate([plabels.copy(), nlabels.copy()], axis=0)
    #             test_dataset.filenames = np.concatenate([pfilenames.copy(), nfilenames.copy()], axis=0)

    #             self.logger.info(f'Test sub dataset: {test_dataset.set_group_flag()}')
    #             test_dataloader = build_dataloaders(cfg.data.test_loader, test_dataset)

    #             preds, labels = self.test(test_dataloader)

    #             pinds = np.argsort(preds[:len(plabels)])
    #             ninds = np.argsort(-preds[len(plabels):])

    #             phn, nhn = int(hr * len(plabels)), int(hr * len(nlabels))
    #             phs, nhs = preds[pinds[: phn]].mean(), preds[ninds[: nhn] + len(plabels)].mean()

    #             # hard_infos = pinfos[pinds[: phn]].tolist() + ninfos[ninds[: nhn]].tolist()
    #             # train_dataset.data_infos.extend(hard_infos)

    #             train_dataset.lmks = np.concatenate([train_dataset.lmks.copy(), 
    #                                         plmks[pinds[: phn]].copy(), nlmks[ninds[: nhn]].copy()], axis=0)
    #             train_dataset.labels = np.concatenate([train_dataset.labels.copy(), 
    #                                         plabels[pinds[: phn]].copy(), nlabels[ninds[: nhn]].copy()], axis=0)
    #             train_dataset.filenames = np.concatenate([train_dataset.filenames.copy(), 
    #                                         pfilenames[pinds[: phn]].copy(), nfilenames[ninds[: nhn]].copy()], axis=0)

    #             self.logger.info(f'Train sub dataset: {train_dataset.set_group_flag()}')
    #             dataloader = build_dataloaders(cfg.data.train_loader, train_dataset)

    #             # pinfos, ninfos = pinfos[pinds[phn:]], ninfos[ninds[nhn:]]
    #             plmks, nlmks =  plmks[pinds[phn:]].copy(), nlmks[ninds[nhn:]].copy()
    #             plabels, nlabels = plabels[pinds[phn:]].copy(), nlabels[ninds[nhn:]].copy()
    #             pfilenames, nfilenames = pfilenames[pinds[phn:]].copy(), nfilenames[ninds[nhn:]].copy()

    #             hr *= dr

    #             self.logger.info(f'Phard num: {phn}, score: {phs}, Nhard num: {nhn}, score: {nhs}')

    #         if self.log_cfg.plog_cfg is not None:
    #             self.vis_log(self.log_cfg.filename)

    #         self.scheduler.step()
    #         self._save_model(filename='latest.pth')

    #     self.logger.info('End of step training!')
