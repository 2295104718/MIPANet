# MIPANet
Optimizing RGB-D Semantic Segmentation through Multi-modal Interactionand Pooling Attention

## Experiment result
we evaluate the mIoU of MIPANet in NYUDv2 and SUN RGB-D

|    | NYUDv2  | SUN RGB-D |
|----|---------|-----------|
| mIoU | 52.3% | 49.1%     |

## Setup
**Requisites:** torch, torchvision, tensorboard, numpy, scipy, tqdm, addict. \
Prepare NYUDv2 and SUN-RGBD datasets:
```
# dataset dir: /[root_path]/data
```
Clone the project repo:
```
# project dir: /[root_path]/MIPANet
git clone https://github.com/2295104718/MIPANet.git
cd MIPANet
```

#### NYU Depth V2

You could download the official NYU Depth V2 data [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). After downloading the official data, you should modify them according to the structure of directories we provide.

#### SUN RGB-D
You could download the official SUN RGB-D data [here](https://rgbd.cs.princeton.edu/). After downloading the official data, you should modify them according to the structure of directories we provide.

## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `tqdm` for showing the training progress. PyTorch v1.1 is supported (using the new supported tensoboard); can work with ealier versions, but instead of using tensoboard, use tensoboardX.

```bash
pip install -r requirements.txt
```

## Training

1) On a computer with GPU:
```
# remember to activate the env
python train.py
```

## Model setting
See model config template under *config.py*



