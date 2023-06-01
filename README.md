

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
All model parameter files of this experiment are in the network disk, and the [link](https://pan.baidu.com/s/1R-BC3i9zzdl1n4k_R6SXnw) is as follows.

# Citation
```

```
