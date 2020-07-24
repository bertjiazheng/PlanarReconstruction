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
from utils.bin_mean_shift import Bin_Mean_Shift
from utils.loss import *
from utils.metric import eval_iou, eval_plane_prediction
from utils.misc import AverageMeter

ex = Experiment('test')


@ex.automain
def test(_run, _config, _log):
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

    save_dir = os.path.join(os.path.dirname(cfg.resume_path), 'predicts')
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
        HoliCityDataset(split='test', transform=transforms, root_dir=cfg.root_dir))

    h, w = 512, 512

    mean_shift = Bin_Mean_Shift(device=device)

    with torch.no_grad():

        for sample in tqdm(data_loader, desc='Test'):
            splits = sample['prefix'][0].split('/')
            datename, filename = splits[-2], splits[-1]

            image = sample['image'].to(device)

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

            predicts = segmentation.cpu().numpy().argmax(axis=1)
            predicts += 1
            predicts[prob.cpu().numpy().reshape(-1) <= 0.1] = 0
            predicts = predicts.reshape(h, w)

            weight_matrix = F.normalize(sample_segmentation, p=1, dim=0)
            instance_params = torch.matmul(sample_params, weight_matrix).t().cpu().numpy()

            if not os.path.exists(os.path.join(save_dir, datename)):
                os.makedirs(os.path.join(save_dir, datename))

            cv2.imwrite(os.path.join(save_dir, datename, f'{filename}_plan.png'), predicts)
            np.savez(os.path.join(save_dir, datename, f"{filename}_plan.npz"),
                ws=instance_params, scores=np.ones(len(instance_params)))
