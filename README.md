# Fine-grained ship recognition for complex background based on global to local and progressive learning
## Requirement
python 3.8

Pytorch >=1.7

torchvision >=0.8

## Training

1. Download datatsets for GLPM (e.g. MAR-ships, CIB-ships, game-of-ships etc) and organize the structure as follows:

dataset

├── train

│   ├── class_001

|   |      ├── 1.jpg

|   |      └── ...

│   ├── class_002

|   |      ├── 1.jpg

└── test

    ├── class_001
    
    |      ├── 1.jpg    
    |      ├── 2.jp
    |      └── ...    
    ├── class_002
    
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...

2、Train from scratch with train.py.
   


if you  use this code, please cite the paper 

"Meng H, Tian Y, Ling Y, et al. Fine-grained ship recognition for complex background based on global to local and progressive learning[J]. IEEE Geoscience and Remote Sensing Letters, 2022, 19: 1-5."

Train

python trian.py

MAR-ships dataset link:

https://blog.csdn.net/u014426939/article/details/104749763?utm_medium=distribute.pc_relevant_download.none-task-blog-baidujs-2.nonecase&depth_1-utm_source=distribute.pc_relevant_download.none-task-blog-baidujs-2.nonecase

paper:Bloisi D D, Iocchi L, Pennisi A, et al. ARGOS-Venice boat classification[C]//2015 12th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS). IEEE, 2015: 1-6.
