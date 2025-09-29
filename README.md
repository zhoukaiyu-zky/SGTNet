# SGTNet
# Experiment
## Environment
```txt
python==3.8.10
pytorch==1.12.1
torchvision==0.13.1
mmengine==0.7.3
mmcv==2.0.0
mmsegmentation==1.0.0
```
## Install
Please refer to mmsegmentation for installation.
## Dataset
SGTNet
```txt
├── mmsegmentation
├── figures
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── CamVid
│   │   ├── train
│   │   ├── train_labels
│   │   ├── test
│   │   ├── test_labels
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
├── SGTNet_cityscapes-1024x1024.py
├── train.py
├── test.py
```
Cityscapes could be downloaded from here. Camvid could be downloaded from here. Pascal VOC 2012 could be downloaded from here.
## Training
If you want to train the SGTNet network on the Cityscapes dataset, please use the following instructions:
Single gpu for train:
```txt
CUDA_VISIBLE_DEVICES=0 python ./mmsegmentation/tools/train.py rdrnet-s-simple_2xb6-120k_cityscapes-1024x1024.py --work-dir ./weight/seg
```
Multiple gpus for train:
```txt
CUDA_VISIBLE_DEVICES=0,1 bash ./mmsegmentation/tools/dist_train.sh rdrnet-s-simple_2xb6-120k_cityscapes-1024x1024.py 2 --work-dir ./weight/seg
```
Train in pycharm: If you want to train in pycharm, you can run it in train.py.
see more details at mmsegmentation.
## Testing
```txt
CUDA_VISIBLE_DEVICES=0,1 bash ./mmsegmentation/tools/dist_train.sh rdrnet-s-simple_2xb6-120k_cityscapes-1024x1024.py 2 --work-dir ./weight/seg
```
Test in pycharm: If you want to test in pycharm, you can run it in test.py.
see more details at mmsegmentation.
## Usage
To visualize your model, go to show_cityscapes.py.

To see the model definitions and do some speed tests, go to MSDSeg.py.

To train, validate, benchmark, and save the results of your model, go to train.py.
## Citation
If you find our work helpful, please consider citing our paper.
```txt
@article{wang2025msdseg,
  title={Lightweight and Real-Time Semantic Segmentation Network with MultiScale Dilated Convolutions},
  author={Shan Zhao, Yunlei Wang, Zhanqiang Huo, Fukai Zhang},
  journal={The Visual Computer},
  year={2025}
}
```
