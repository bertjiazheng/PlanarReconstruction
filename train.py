import os
import pickle
import random
import time
from tqdm import trange, tqdm

import cv2
import numpy as np
from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.utils.tensorboard import SummaryWriter

from models.fpn import FPN
from dataset import HoliCityDataset
from utils.bin_mean_shift import Bin_Mean_Shift
from utils.loss import *
from utils.metric import eval_iou, eval_plane_prediction
from utils.misc import AverageMeter, get_optimizer


ex = Experiment('train')
ex.add_config('./configs/config.yaml')


@ex.automain
def train(_run, _config, _log):
    cfg = edict(_config)

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        _log.info("Let's use GPU!")
    else:
        device_name = "cpu"
        _log.info("CUDA is not available")
    device = torch.device(device_name)

    # add observers and tensorboard log
    if not _run.unobserved:
        tb_writer = SummaryWriter(log_dir=_run.observers[0].dir)
        checkpoint_dir = _run.observers[0].dir

    # build network
    network = FPN(cfg.model)
    network.to(device)
    network.train()

    # set up optimizers
    optimizer = get_optimizer(network.parameters(), cfg.solver)

    # data loader
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_loader = torch.utils.data.DataLoader(
        HoliCityDataset(split='train', transform=transforms, root_dir=cfg.root_dir),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

    mean_shift = Bin_Mean_Shift(device=device)
    ins_param_loss = InstanceParameterLoss()
    ins_param_loss.to(device)

    iteration = 0
    # main loop
    for epoch in trange(cfg.num_epochs, desc='Epoch'):

        for sample in tqdm(data_loader, desc='Train', position=1, leave=False):

            losses = AverageMeter()
            losses_pull = AverageMeter()
            losses_push = AverageMeter()
            losses_cls = AverageMeter()
            losses_depth = AverageMeter()
            losses_normal = AverageMeter()
            losses_instance = AverageMeter()

            ioues = AverageMeter()
            mean_angles = AverageMeter()
            rmses = AverageMeter()
            instance_rmses = AverageMeter()

            for key in sample:
                sample[key] = sample[key].to(device)

            # forward pass
            logit, embedding, param = network(sample['image'])

            segmentations, sample_segmentations, sample_params, _, _, _ = \
                mean_shift(logit, embedding, param, sample['segmentation'])

            # calculate loss
            for i in range(cfg.batch_size):
                loss, loss_pull, loss_push = hinge_embedding_loss(
                    embedding[i], sample['instance'][i], sample['num_planes'][i])

                loss_cls = class_balanced_cross_entropy_loss(
                    logit[i], sample['planar_region'][i])

                loss_normal, mean_angle = surface_normal_loss(
                    param[i], sample['plane_parameters'][i], sample['planar_region'][i])

                loss_l1 = plane_parameter_loss(
                    param[i], sample['plane_parameters'][i], sample['planar_region'][i])

                loss_depth, rmse, _ = Q_loss(
                    param[i], ins_param_loss.k_inv_dot_xy1, sample['depth'][i])

                if segmentations[i] is None:
                    continue

                loss_instance, _, instance_abs_disntace, _ = ins_param_loss(
                    segmentations[i], sample_segmentations[i], sample_params[i],
                    sample['planar_region'][i], sample['depth'][i])

                loss += loss_cls + loss_normal + loss_l1 + loss_depth + loss_instance

                # planar segmentation iou
                prob = torch.sigmoid(logit[i])
                mask = (prob > 0.5).float().cpu().numpy()
                iou = eval_iou(mask, sample['planar_region'][i].cpu().numpy())

                ioues.update(iou * 100)
                mean_angles.update(mean_angle.item())
                rmses.update(rmse.item())
                instance_rmses.update(instance_abs_disntace.item())

                losses.update(loss.item())
                losses_pull.update(loss_pull.item())
                losses_push.update(loss_push.item())
                losses_cls.update(loss_cls.item())
                losses_depth.update(loss_depth.item())
                losses_normal.update(loss_normal.item())
                losses_instance.update(loss_instance.item())

            loss = loss / cfg.batch_size

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not _run.unobserved:
                tb_writer.add_scalar('loss', losses.avg, iteration)
                tb_writer.add_scalar('loss/pull_loss', losses_pull.avg, iteration)
                tb_writer.add_scalar('loss/push_loss', losses_push.avg, iteration)
                tb_writer.add_scalar('loss/cls_loss', losses_cls.avg, iteration)
                tb_writer.add_scalar('loss/normal_loss', losses_normal.avg, iteration)
                tb_writer.add_scalar('loss/depth_loss', losses_depth.avg, iteration)
                tb_writer.add_scalar('loss/instance_loss', losses_instance.avg, iteration)

                tb_writer.add_scalar('accu/ioues', ioues.avg, iteration)
                tb_writer.add_scalar('accu/angles', mean_angles.avg, iteration)
                tb_writer.add_scalar('accu/rmes', rmses.avg, iteration)
                tb_writer.add_scalar('accu/ins_rmse', instance_rmses.avg, iteration)

            iteration+= 1

        # save checkpoint
        if not _run.unobserved:
            torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))
