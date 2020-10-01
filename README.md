# Deep Coded Aperture Design: An End-to-End Approach for Computational Imaging Tasks
This repo provides a the python code of the paper "Deep Coded Aperture Design: An End-to-EndApproach for Computational Imaging Tasks"

# Install

The list of some libraries that you need to execute the code:
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
or using pip install and the requirement file.

# Data
This work was based on three datasets; please download the datasets and put it correctly (Train/Test) in the dataset folder.
- *MNIST dataset*: Provided in the `dataset/MNIST` folder.
- [*ARAD hyperspectral dataset:*](https://competitions.codalab.org/competitions/22225) 450 hyperspectral training images and 10 validation images. The dataset  are available on the [challenge track websites](https://competitions.codalab.org/competitions/22225), registration is required to access data.
- [*NYU Depth Dataset:*](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) contains $1449$ RGB images we use a depth map of 15 discretization levels and its semantic labels for 13 classes. A Matlab function to convert to 15 discretization levels is provided in the `dataset/NYU` folder. 


## Structure of directories

| directory  | description  |
| :--------: | :----------- | 
| `algorithms` | MATLAB functions of main algorithms proposed in the paper (original) | 
| `tests`    | MATLAB scripts to reproduce the results in the paper (original) |
| `packages`   | algorithms adapted from the state-of-art algorithms (adapted)|
| `dataset`    | data used for reconstruction (simulated and real data, refer to the readme file for details) |
| `results`    | results of reconstruction (after reconstruction) |
| `utils`      | utility functions |


