import os

import cv2
import numpy as np
from tqdm import tqdm


def compute_plane_nums(root_dir):
    folderlist = sorted(os.listdir(root))

    for folder in folderlist:

        if '.txt' in folder:
            continue

        for filename in os.listidr(os.path.join(root, folder)):

            path = os.path.join(root, folder, filename)

            if path.endswith('_plan.npz'):
                img_path = path.replace('.npz', '.png')
                img = cv2.imread(img_path, -1)

                with np.load(path) as N:
                    plane_normal = N["ws"]

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(5, 3))
    # num_planes = np.load('./data/num_planes.npy')

    # ax.plot(np.arange(len(num_planes)) + 1, num_planes)

    # ax.autoscale(tight=True)
    # ax.grid()

    # plt.legend(loc="lower right")
    # plt.show()


def filter_small_planes(root_dir='/HDDData/jiajia/HoliCity', split='train', max_planes=100):

    filelist = np.genfromtxt(f"{root_dir}/filelist.txt", dtype=str)
    filter_ = np.genfromtxt(f"{root_dir}/{split}-middlesplit.txt", dtype=str)
    length = len(filter_[0])
    filter_ = set(filter_)

    filelist = [f"{root_dir}/{f}" for f in filelist if f[:length] in filter_]

    for prefix in tqdm(filelist):
        if not os.path.exists(f"{prefix}_plan.png") and not os.path.exists(f"{prefix}_plan.npz"):
            print(prefix)
            continue

        plane_mask = cv2.imread(f"{prefix}_plan.png", -1)
        with np.load(f"{prefix}_plan.npz") as N:
            plane_normal = N["ws"]

        indices, counts = np.unique(plane_mask, return_counts=True)

        orders = indices[np.argsort(counts)[::-1]]

        plane_mask_new = np.zeros_like(plane_mask)
        plane_normal_new = []

        new_index = 1
        for old_index in orders:
            if old_index == 0:
                continue
            if new_index > max_planes:
                break
            plane_mask_new[plane_mask == old_index] = new_index
            plane_normal_new.append(plane_normal[old_index - 1])
            new_index += 1

        if len(plane_normal_new) == 0:
            print(prefix, len(plane_normal))
            continue

        plane_normal_new = np.stack(plane_normal_new)

        # save
        cv2.imwrite(f'{prefix}_plan_v1.png', plane_mask_new)
        np.savez(f"{prefix}_plan_v1.npz", ws=plane_normal_new)
    
    filelist = np.genfromtxt(f"{root_dir}/filelist.txt", dtype=str)
    filelist = [filename for filename in filelist if os.path.exists(os.path.join(root_dir, f'{root_dir}/{filename}_plan_v1.png'))]
    np.savetxt(f"{root_dir}/filelist_v1.txt", filelist, fmt="%s")


if __name__ == "__main__":
    filter_small_planes()
