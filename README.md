# Pix2Pix-Pytorch


## Introduction
Translation of city satellite maps based on Pix2Pix


![Result](./results/Result.gif)

## Environment
The code is developed using python 3.7 on Windows 10. NVIDIA GPUs are needed. 
The code is developed and tested using NVIDIA RTX2070 GPU card. Other platforms or GPU cards are not fully tested.

## Quick start

### Installation
 1. Please follow the dependencies below:
 ```
 torch >= 1.4.1
 torchvision
 tqdm
 numpy
 PIL
 ```

 
### Data preparation
 1. Download the Maps dataset in the link
 [Pix2Pix Maps](https://www.kaggle.com/alincijov/pix2pix-maps)
 2. Put the dataset into this directory, your directory tree should look like this:
 
 ```
 |--data
    |--train
        |--1.jpg
        |--2.jpg
        ···
    |--val
        |--1.jpg
        |--2.jpg
        ···
 |--results
 |--util
 |--train.py
 |--README.md
 ```
 
 ### Training
Just enter the following commands in the command line to start training the Pix2Pix model, and the training results and model will be automatically saved
```
python train.py
```

If you want to customize some basic parameters, you can use the following command to query
```
python train.py -h
```

### Result
The following is the result of the program running 99 epochs by default

![y6hLHf.png](https://s3.ax1x.com/2021/02/15/y6hLHf.png)
