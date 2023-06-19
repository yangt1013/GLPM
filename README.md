# Fine-Grained Ship Recognition for Complex Background Based on Global to Local and Progressive Learning
## Requirement
python 3.8

Pytorch >=1.7

torchvision >=0.8

## Training

1. Download datatsets for GLPM (e.g. MAR-ships, CIB-ships, game-of-ships etc) and organize the structure as follows:
```bash
dataset

└── train/test

    ├── class_001
    
    |      ├── 1.jpg    
    |      ├── 2.jp
    |      └── ...    
    ├── class_002
    
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```
2、Train from scratch with `train.py`.
## Citation
Please cite our paper if you use GLPM in your work.
```bash
@InProceedings{du2021fine,
  title={Fine-Grained Ship Recognition for Complex Background Based on Global to Local and Progressive Learning},
  author={Hao Meng; Yang Tian; Yue Ling; Tao Li},
  booktitle = {IEEE Geoscience and Remote Sensing Letters},
  year={2021}
}
```
MAR-ships dataset link:
```bash
https://blog.csdn.net/u014426939/article/details/104749763?utm_medium=distribute.pc_relevant_download.none-task-blog-baidujs-2.nonecase&depth_1-utm_source=distribute.pc_relevant_download.none-task-blog-baidujs-2.nonecase
```
```bash
paper:Bloisi D D, Iocchi L, Pennisi A, et al. ARGOS-Venice boat classification[C]//2015 12th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS). IEEE, 2015: 1-6.
```
