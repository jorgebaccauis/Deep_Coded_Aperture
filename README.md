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

## Structure of directories

| directory  | description  |
| :--------: | :----------- | 
| `algorithms` | MATLAB functions of main algorithms proposed in the paper (original) | 
| `tests`    | MATLAB scripts to reproduce the results in the paper (original) |
| `packages`   | algorithms adapted from the state-of-art algorithms (adapted)|
| `dataset`    | data used for reconstruction (simulated and real data, refer to the readme file for details) |
| `results`    | results of reconstruction (after reconstruction) |
| `utils`      | utility functions |


