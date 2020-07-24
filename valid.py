import os
import pickle
import random
import time
from tqdm import tqdm

import cv2
import numpy as np
from easydict import EasyDict as edict
from sacred import Experiment

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

from models.fpn import FPN
from dataset import HoliCityDataset
from utils.disp import labelcolormap, tensor_to_image
from utils.bin_mean_shift import Bin_Mean_Shift
from utils.loss import *
from utils.metric import eval_iou, eval_plane_prediction
from utils.misc import AverageMeter

ex = Experiment('valid')


@ex.automain
def valid(_run, _config, _log):
    cfg = edict(_config)

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        _log.info("Let's use GPU!")
    else:
        device_name = "cpu"
        _log.info("CUDA is not available")
    device = torch.device(device_name)

    # build network
    network = FPN(cfg.model)

    if os.path.exists(cfg.resume_path):
        _log.info(f"Load model from {cfg.resume_path}")
        model_state = torch.load(cfg.resume_path, map_location='cpu')
        network.load_state_dict(model_state)
    else:
        _log.info(f"{cfg.resume_path} not found!")
        exit()

    save_dir = os.path.join(os.path.dirname(cfg.resume_path), 'valid')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load nets into gpu
    network.to(device)
    network.eval()

    # data loader
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_loader = torch.utils.data.DataLoader(
        HoliCityDataset(split='valid', transform=transforms, root_dir=cfg.root_dir))

    h, w = 512, 512

    mean_shift = Bin_Mean_Shift(device=device)

    colormap = labelcolormap(256)

    with torch.no_grad():

        for sample in tqdm(data_loader, desc='valid'):
            splits = sample['prefix'][0].split('/')
            datename, filename = splits[-2], splits[-1]

            image = sample['image'].to(device)
            plane_mask = sample['segmentation'][0]
            num_planes = sample['num_planes'].numpy()[0]

            # forward pass
            logit, embedding, param = network(image)

            prob = torch.sigmoid(logit[0])

            # fast mean shift
            segmentation, sample_segmentation, sample_params = mean_shift.test_forward(
                prob, embedding[0], param, mask_threshold=0.1)

            # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned,
            # we use avg_pool_2d to smooth the segmentation results
            b = segmentation.t().view(1, -1, h, w)
            b_pooling = F.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
            segmentation = b_pooling.view(-1, h * w).t()

            pred_mask = segmentation.cpu().numpy().argmax(axis=1)
            pred_mask += 1

            pred_mask[prob.cpu().numpy().reshape(-1) <= 0.1] = 0
            pred_mask = pred_mask.reshape(h, w)

            # visualize
            rgb_viz = tensor_to_image(image[0].cpu())
            pred_mask_viz = np.stack([colormap[pred_mask, 0], colormap[pred_mask, 1], colormap[pred_mask, 2]], axis=2)
            gt_mask_viz = np.stack([colormap[plane_mask, 0], colormap[plane_mask, 1], colormap[plane_mask, 2]], axis=2)
            image_viz = np.concatenate((rgb_viz, pred_mask_viz, gt_mask_viz), axis=1)
            if not os.path.exists(os.path.join(save_dir, datename)):
                os.makedirs(os.path.join(save_dir, datename))
            cv2.imwrite(os.path.join(save_dir, datename, f'{filename}.png'), image_viz)
