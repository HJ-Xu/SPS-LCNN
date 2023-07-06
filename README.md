

# SPS-LCNN

Code for "SPS-LCNN: A Significant Point Sampling-Based Lightweight Convolutional Neural Network for Point Cloud Processin" 


# Installation
```
python = 3.6
cuda = 9.0
TensorFlow = 1.8.0
Ubuntu = 16.04
```

# Download data
Please put the downloaded [data](https://drive.google.com/file/d/1GGtRzEWXfWUPGnYpdzu9TkP6vfYzDmnV/view?usp=drive_link) in the off format into the directory `data/.../`.

# Compile Customized TF Operators
The TF operators are included under 'tf_ops/', you need to compile them 'sh xxx.sh' (check under each ops subfolder) first. Update and path if necessary.

# Usage
To train SPS-LCNN, use the training script:
```
> python train.py  
```
We provide networks trained on the modelnet40 datasets, and the network parameters are saved in `log/` . You can directly run the following code to verify the experimental accuracy mentioned in the paper:
```
> python evaluate.py
```
The SPS module is in the `SPS-LCNN/utils/pointnet_util.py/ECA_model`

All model parameter files of this experiment are in the network disk, and the [link](https://pan.baidu.com/s/1R-BC3i9zzdl1n4k_R6SXnw) is as follows.

# Citation
```
@article{XU2023110498,
title = {SPS-LCNN: A Significant Point Sampling-based Lightweight Convolutional Neural Network for point cloud processing},
journal = {Applied Soft Computing},
volume = {144},
pages = {110498},
year = {2023},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2023.110498},
url = {https://www.sciencedirect.com/science/article/pii/S1568494623005161},
author = {Haojun Xu and Jing Bai},
keywords = {Lightweight, Point clouds, Significant Point Sampling, Convolutional Neural Networks},
abstract = {Point cloud data have very promising applications, but the irregularity and disorder make it a challenging problem how to use them. In recent years, an increasing number of new and excellent research solutions have been proposed, which focus on exploring local feature extractors. Over-engineered feature extractors lead to saturating the performance of current methods and often introduce unfavorable latency and additional overhead. This defeats the original purpose of using point cloud data, which is simplicity and efficiency. In this paper, we construct a learnable pipeline by designing two core modules with a small number of parameters – significant point sampling (SPS) and multiscale significant feature extraction (MS-SFE) – to balance accuracy and overhead. Our pipeline demonstrates comparable performance to state-of-the-art methods while requiring fewer parameters, making it well-suited for real-time applications.}
}
```
