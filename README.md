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
ARGOS-Venice boat classification

website：https://pan.baidu.com/s/1OHBMLMXvkKima1nK5gF-4A?pwd=GLPM 
word：GLPM 
```
```bash
game-of-ships dataset link:

website：https://pan.baidu.com/s/1XkSwtPnxKblxZ6YQV4tEDA?pwd=GLPM 
word：GLPM 
```
