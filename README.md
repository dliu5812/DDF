# Decompose to Adapt: Cross-domain Object Detection via Feature Disentanglement


## Introduction

In this project, we proposed a Domain Disentanglement Faster-RCNN (DDF) for cross-domain object detection, from the view of feature disentanglement.

The implementations are for our paper published in IEEE Transactions on Multimedia:

[Decompose to Adapt: Cross-domain Object Detection via Feature Disentanglement](https://arxiv.org/abs/2201.01929)

![alt text](figs/ddf-overall-fig.png "overall framework")

## Preparation

### Basic settings

* Python 3+
* Pytorch 1.6.0
* CUDA 11.0

### Dataset Preparation

* [Cityscapes & Foggy Cityscapes](https://www.cityscapes-dataset.com/)
  (You can also download the dataset [GoogleDrive](https://drive.google.com/file/d/1mA0L5-1U_Vo-S8-cv12QBmhgG9FFf6nf/view?usp=sharing))
* [SIM 10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/.

### Compilation

You might need to re-build this repository via:

```
cd lib  
python setup.py build develop
```

For other detailed settings, please refer to pytorch 1.0 version of [repository](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0).

## Training

To train the model, please run:

```
./train_disent.sh
```

## Test and visualization

To test the model, please run:

```
./test_disent.sh
```


To get the visualization of the feature maps and the feature distance, please refer to:

```
./test_disent_vis.sh
```


## Visual Examples of the features

![alt text](figs/ddf-intro-fig.jpg "example images")



## Citations (Bibtex)
Please consider citing our papers in your publications if they are helpful to your research:
```
@article{liu2022decompose,
  title={Decompose to Adapt: Cross-domain Object Detection via Feature Disentanglement},
  author={Liu, Dongnan and Zhang, Chaoyi and Song, Yang and Huang, Heng and Wang, Chenyu and Barnett, Michael and Cai, Weidong},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}

```


## Contact

Please contact Dongnan Liu (dongnanliu0201@gmail.com) regarding any issues.


## License

DDF is released under the MIT license. See [LICENSE](LICENSE) for additional details.