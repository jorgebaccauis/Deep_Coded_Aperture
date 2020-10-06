# Deep Coded Aperture Design: An End-to-End Approach for Computational Imaging Tasks
This repository provides the Python source codes related to the paper "Deep Coded Aperture Design: An End-to-EndApproach for Computational Imaging Tasks"

# Installation

List of libraries required to execute the code.:
- python = 3.7.7
- Tensorflow = 2.2
- Keras = 2.4.3
- numpy
- scipy
- matplotlib
- h5py = 2.10
- opencv = 4.10
- poppy = 0.91

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```
or using pip install and the required file.

# Data
This work uses the following three datasets. Please download the datasets and store them it correctly in the corresponding dataset folder (Train/Test).
- *MNIST dataset*: Provided in the `dataset/MNIST` folder.
- [*ARAD hyperspectral dataset:*](https://competitions.codalab.org/competitions/22225) It contains 450 hyperspectral training images and 10 validation images. The dataset  is available on the [challenge track websites](https://competitions.codalab.org/competitions/22225). Note that registration is required to access data.
- [*NYU Depth Dataset:*](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) It contains 1449 RGB images. We use a depth map of 15 discretization levels and its semantic labels for 13 classes. A Matlab function to convert to 15 discretization levels is provided in the `dataset/NYU` folder. 


## Structure of directories

| Directory  | Description  |
| :--------: | :----------- | 
| `Dataset` | Folder that contains the datasets | 
| `Models and Tools`    | `.py` files for the custumer models |



