import os

import cv2
import numpy as np

import torch


class HoliCityDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, root_dir=None,
                 max_num_planes=100):
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.transform = transform
        self.root_dir = root_dir

        self.max_num_planes = max_num_planes

        if split == 'train':
            filelist = np.genfromtxt(f"{root_dir}/filelist_v1.txt", dtype=str)
        else:
            filelist = np.genfromtxt(f"{root_dir}/filelist.txt", dtype=str)

        if split == 'test':
            # submit the results of the valid and test splits
            filter_1 = np.genfromtxt(f"{root_dir}/valid-middlesplit.txt", dtype=str)
            filter_2 = np.genfromtxt(f"{root_dir}/test-middlesplit.txt", dtype=str)
            filter_ = np.concatenate((filter_1, filter_2))
        else:
            filter_ = np.genfromtxt(f"{root_dir}/{split}-middlesplit.txt", dtype=str)

        length = len(filter_[0])
        filter_ = set(filter_)

        self.filelist = [f"{root_dir}/{f}" for f in filelist if f[:length] in filter_]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        prefix = self.filelist[index]

        # load data
        image = cv2.imread(f"{prefix}_imag.jpg", -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.split == 'test':
            sample = {
                'prefix': prefix,
                'image': image
            }
            return sample

        if self.split == 'train':
            plane_mask = cv2.imread(f"{prefix}_plan_v1.png", -1)
            plane_mask = plane_mask.astype(np.int)

            with np.load(f"{prefix}_plan_v1.npz") as N:
                plane_normal = N["ws"]

            num_planes = len(plane_normal)
            plane_mask_oh = np.zeros([self.max_num_planes, 512, 512], dtype=np.bool)

        elif self.split == 'valid':
            plane_mask = cv2.imread(f"{prefix}_plan.png", -1)
            plane_mask = plane_mask.astype(np.int)

            with np.load(f"{prefix}_plan.npz") as N:
                plane_normal = N["ws"]

            num_planes = len(plane_normal)
            plane_mask_oh = np.zeros([num_planes, 512, 512], dtype=np.bool)

        plane_params = np.zeros((3, 512, 512))
        depth = np.zeros((512, 512), dtype=float)

        x, y = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(1, -1, 512))
        xyz = np.stack([x, y, -np.ones_like(x)], axis=2)

        valid_region = plane_mask != 0

        for i in range(num_planes):
            # 0 indicates non-planar region
            mask = plane_mask == (i+1)
            plane_mask_oh[i, mask] = 1
            plane_params[:, mask] = plane_normal[i].reshape(-1, 1)
            depth[mask] = 1 / (xyz[mask] @ plane_normal[i])
        depth[depth < 0] = 0
        depth[depth > 1000] = 1000

        sample = {
            'prefix': prefix,
            'image': image,
            'num_planes': num_planes,
            'instance': torch.from_numpy(plane_mask_oh),            # plane instance one hot
            'planar_region': torch.from_numpy(valid_region),
            'segmentation': torch.from_numpy(plane_mask),           # plane mask
            'depth': torch.from_numpy(depth).float(),
            'plane_parameters': torch.from_numpy(plane_params).float(),
        }

        return sample
