# HoliCity Plane Detection via Associative Embedding

The repository provides [PlaneAE](http://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Single-Image_Piece-Wise_Planar_3D_Reconstruction_via_Associative_Embedding_CVPR_2019_paper.html) baseline for [HoliCity Plane Detection Track](https://competitions.codalab.org/competitions/24942) on [Holistic 3D Vision Challenge](https://holistic-3d.github.io/eccv20/challenge.html) at ECCV 2020.

### Training
Run the following command to train our network:

```bash
python train.py -F logs with dataset.root_dir=/path/to/HoliCity
```

### Evaluation
Run the following command to evaluate the performance:

```bash
python valid.py -uf with /path/to/config.json resume_path=/path/to/network.pt
```

### Prediction
Run the following command to predict on a single image:
```bash
python test.py -uf with /path/to/config.json resume_path=/path/to/network.pt
```

## Citation

Please cite our paper if it helps your research:

```bibtex
@inproceedings{YuZLZG19,
  author    = {Zehao Yu and Jia Zheng and Dongze Lian and Zihan Zhou and Shenghua Gao},
  title     = {Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding},
  booktitle = {CVPR},
  pages     = {1029--1037},
  year      = {2019}
}
```
