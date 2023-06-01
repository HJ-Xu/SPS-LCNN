""" Point2Sequence Layers

Author: Xinhai Liu
Date: June 2018
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
# from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point
# from tf_interpolate import three_nn, three_interpolate
from sklearn.decomposition import PCA
import tensorflow as tf
import math
import numpy as np
import tf_util
import tensorflow.contrib.slim as slim
from layers import *
import pointfly as pf
from tflearn.layers.conv import global_avg_pool,global_max_pool

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 3+d)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 3+d)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
    
def conv(input_channel, kernel, mlp, stride, is_training, scope, bn_decay, concat=False, concatdata=None,max_pooling=False):
    with tf.variable_scope(scope) as sc:
        if len(input_channel.get_shape().as_list()) !=4:
            input_channel = tf.expand_dims(input_channel,axis=-2)
        # input_channel = tf_util.conv2d(input_channel, mlp2, [1,1],padding='VALID', stride=[1,1], bn=True, is_training=is_training,scope='conv1', bn_decay=bn_decay)
        for i, out_channel in enumerate(mlp):
            if concat:
                if i == len(mlp)-1:
                    input_channel = tf.concat([input_channel,concatdata], axis = -1)
            net = tf_util.conv2d(input_channel, out_channel, kernel,padding='VALID', stride=stride, bn=True, is_training=is_training,scope='conv%d'%(i), bn_decay=bn_decay)
            # net = tf.nn.atrous_conv2d(input_channel, kernel, rate=16, padding="VALID")
            input_channel = net
        if max_pooling:
            net = tf.squeeze(tf.reduce_max(net, axis=[1], keep_dims=True))
        # gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        # net = gamma*net+net

    return net


def dilated_conv2D_layer(inputs,num_outputs,kernel_size,is_training,bn_decay,rate,padding,scope,data_format='NHWC',bn=True,use_xavier=True,stddev=1e-3,weight_decay=None,activation_fn=tf.nn.relu,weights_regularizer=slim.l1_regularizer(scale=0.01)):
    with  tf.variable_scope(name_or_scope=scope):
        in_channels = inputs.get_shape().as_list()[3]
        kernel=[kernel_size[0],kernel_size[1],in_channels,num_outputs]

        # # filter_weight=tf.Variable(initial_value=tf.truncated_normal(shape,stddev=0.1))\
        # filter_weight = slim.variable(name='weights',
        #                               shape=kernel,
        #                               initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                               regularizer=weights_regularizer)
        # bias = tf.Variable(tf.constant(0.01, shape=[num_outputs]))
        # # inputs = tf.nn.conv2d(inputs,filter_weight, strides, padding=padding) + bias
        # inputs = tf.nn.atrous_conv2d(inputs, filter_weight, rate=rate, padding=padding) + bias

        filter_weight = tf_util._variable_with_weight_decay('weights',
                                             shape=kernel,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)

        outputs = tf.nn.atrous_conv2d(inputs, filter_weight, rate=rate, padding=padding)

        biases = tf_util._variable_on_cpu('biases', [num_outputs],tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
          outputs = tf_util.batch_norm_for_conv2d(outputs, is_training,
                                          bn_decay=bn_decay, scope='bn',
                                          data_format=data_format)

        if not activation_fn is None:
            outputs = activation_fn(outputs)

        return outputs

def separable_conv2d(input, output, name, is_training, kernel_size, strides, depth_multiplier=1,
                     reuse=None, with_bn=True, activation=tf.nn.elu):
    conv2d = tf.layers.separable_conv2d(input, output, kernel_size=kernel_size, strides=strides, padding='VALID',
                                        activation=activation,
                                        depth_multiplier=depth_multiplier,
                                        depthwise_initializer=tf.glorot_normal_initializer(),
                                        pointwise_initializer=tf.glorot_normal_initializer(),
                                        depthwise_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                        pointwise_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                        reuse=reuse, name=name, use_bias=not with_bn)
    return batch_normalization(conv2d, is_training, name + '_bn', reuse) if with_bn else conv2d

def batch_normalization(data, is_training, name, reuse=None):
    return tf.layers.batch_normalization(data, momentum=0.99, training=is_training,
                                         beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         reuse=reuse, name=name)

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,new_points.get_shape()[2].value],
                                        padding='VALID', stride=[1,new_points.get_shape()[2].value],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,new_points.get_shape()[2].value],
                                            padding='VALID', stride=[1,new_points.get_shape()[2].value],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def attention_select_point(x, mlp, xyz, name, is_training, bn_decay, pointtop, pointbom):
    with tf.variable_scope(name):
        if len(x.get_shape().as_list()) !=4:
            x = tf.expand_dims(x,axis=-2)

        batch_size, npoints, kpoints, num_channels = x.get_shape().as_list()
        q = tf_util.conv2d(x, mlp, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='q', bn_decay=bn_decay)

        k = tf_util.conv2d(x, mlp, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='k', bn_decay=bn_decay)

        # v = tf_util.conv2d(x, mlp, [1,1],
        #                                 padding='VALID', stride=[1,1],
        #                                 bn=True, is_training=is_training,
        #                                 scope='v', bn_decay=bn_decay)

        s = tf.matmul(hw_flatten(q), hw_flatten(k), transpose_b=True) # # [bs, N, N]
        s = tf.nn.softmax(s)  # attention map
        # w = tf.get_variable('weights', (s.shape), tf.float32, initializer=tf.constant_initializer(1.0))
        # s = w*s+gamma

        beta = tf.reduce_sum(s,axis=-1,keep_dims=True)
        # gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        # w = tf.get_variable('weights', (beta.shape), tf.float32, initializer=tf.constant_initializer(1.0))
        # beta = w*beta+gamma

        beta = tf.reshape(beta,[batch_size,1,-1])

        _, nn_idx1 = tf.nn.top_k(beta, k=pointtop, sorted=True)
        _, nn_idx2 = tf.nn.top_k(-beta, k=pointbom, sorted=True)
        # _, nn_idx2 = tf.nn.top_k(-beta, k=point//2, sorted=True)
        nn_idx = tf.concat([nn_idx1,nn_idx2],axis=-1)
        nn_idx = tf.reshape(nn_idx,[batch_size,-1,1])

        x = tf.squeeze(x)
        # x = hw_flatten(x)
        feature = find_xyz(x, nn_idx)
        new_xyz = tf.squeeze(find_xyz(xyz, nn_idx))

        # feature = tf.squeeze(get_xyz(x, nn_idx))
        # new_xyz = tf.squeeze(find_xyz(xyz, nn_idx))
    return new_xyz,feature

def Squeeze_excitation_layer(x, xyz, out_dim, layer_name, is_training, bn_decay):
    with tf.name_scope(layer_name) :
        if len(x.get_shape().as_list()) !=4:
            x = tf.expand_dims(x,axis=-2)
        input_x = tf.transpose(x, [0,3,2,1])
        # squeeze = Global_Average_Pooling(input_x)

        squeeze_max = tf.reduce_max(input_x, axis=[1], keep_dims=True, name='maxpool')
        squeeze_mean = tf.reduce_mean(input_x, axis=[1], keep_dims=True, name='avgpool')
        squeeze = squeeze_mean + squeeze_max

        excitation = pf.dense(squeeze, out_dim, layer_name+'_fully_connected1', is_training)
        # excitation = Fully_connected(squeeze, units=out_dim, layer_name=layer_name+'_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = pf.dense(excitation, input_x.get_shape()[-1].value, layer_name+'_fully_connected2', is_training)
        # excitation = Fully_connected(excitation, units=input_x.get_shape()[-1].value, layer_name=layer_name+'_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,excitation.get_shape()[-1].value])

        # scale = input_x * excitation
        # scale = tf.transpose(scale, [0,3,2,1])
        # scale = tf.reduce_sum(scale,axis=-1,keep_dims=True)
        # scale = tf.reshape(scale,[scale.get_shape()[0].value,-1,scale.get_shape()[1].value])

        _, nn_idx = tf.nn.top_k(excitation, k=out_dim, sorted=True)
        nn_idx = tf.reshape(nn_idx,[nn_idx.get_shape()[0].value,-1,1])

        x = tf.squeeze(x)
        feature = find_xyz(x, nn_idx)
        new_xyz = tf.squeeze(find_xyz(xyz, nn_idx))

        return new_xyz,feature

def ECA_model(x, xyz, out_dim, layer_name, is_training, bn_decay):
    with tf.name_scope(layer_name) :
        if len(x.get_shape().as_list()) !=4:
            x = tf.expand_dims(x,axis=-2)
        b, n, kp, c = x.get_shape().as_list()
        input_x = tf.transpose(x, [0,3,2,1])

        # squeeze = Global_Average_Pooling(input_x)

        # squeeze_max = tf.reduce_max(input_x, axis=[1], keep_dims=True, name='maxpool')
        # squeeze_mean = tf.reduce_mean(input_x, axis=[1], keep_dims=True, name='avgpool')
        y_max = global_max_pool(input_x, name='Global_max_pooling')
        # y_mean = global_avg_pool(input_x, name='Global_avg_pooling')
        # y = tf.expand_dims((y_mean + y_max),axis=-1)
        y = tf.expand_dims(y_max,axis=-1)
        # t = int(abs((math.log(n,2)+1)/2))
        # k = t if t%2==0 else t+1
        if n == 1024:
            k = 256
        elif n == 512:
            k = 128

        y = tf.concat([y,y[:,0:k-1,:]],axis=1)
        # SAME VALID
        y = tf_util.ECA(y, 1, k, padding='VALID', stride=1, scope='layer_name', bn=False, is_training=is_training, bn_decay=bn_decay)

        y = tf.transpose(y, [0,2,1])

        y = tf.nn.sigmoid(y)

        # scale = input_x * excitation
        # scale = tf.transpose(scale, [0,3,2,1])
        # scale = tf.reduce_sum(scale,axis=-1,keep_dims=True)
        # scale = tf.reshape(scale,[scale.get_shape()[0].value,-1,scale.get_shape()[1].value])

        _, nn_idx = tf.nn.top_k(y, k=out_dim, sorted=True)
        # nn_idx = tf.reshape(nn_idx,[nn_idx.get_shape()[0].value,-1,1])
        nn_idx = tf.transpose(nn_idx, [0,2,1])

        x = tf.squeeze(x)
        feature = find_xyz(x, nn_idx)
        new_xyz = tf.squeeze(find_xyz(xyz, nn_idx))
        return new_xyz,feature

def point2sequence_module(xyz, points, npoint1, npoint2, nsample_list, mlp_list, output_size, is_training, bn_decay, scope, radi=None, bn=True, use_xyz=True, use_nchw=False, batch_size=16, select=True, num_groups=2, depth_multiplier=4, attention=False):
    ''' Point2sequence module
        assume mlp[k][-1] are all the same for rnn input
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            hidden_size: int32 -- hidden size of the RNN hidden state
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            feature: (batch_size, npoint, output_size}) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        new_points_list = []

        if select:
            # attention
            # new_xyz,new_feature = attention_select_point(points, points.get_shape()[-1].value, xyz, 'asp', is_training, bn_decay, pointtop=npoint1, pointbom=npoint2)

            # ECA
            new_xyz,new_feature = ECA_model(points, xyz, npoint1, 'sqp', is_training, bn_decay)
            points = tf.squeeze(points)

            # fps
            # idx_fps = farthest_point_sample(npoint1, xyz)
            # new_feature = find_xyz(points, tf.expand_dims(idx_fps,axis=-1))
            # new_xyz = gather_point(xyz, idx_fps)
        else:
            feature = xyz
            new_xyz = xyz

        if nsample_list[0] != 0:
            if radi:
                idx_ = pf.radince_indices_general(new_xyz, xyz, radi[0], nsample_list[-1])
            else:
                idx_ = pf.knn_indices_general(new_xyz, xyz, nsample_list[-1], True)
            grouped_xyz = tf.gather_nd(xyz, idx_)
            if points is not None:
                grouped_points = tf.gather_nd(points, idx_)
                sk_grouped_points,_ = tf.nn.moments(grouped_points, axes=2,keep_dims=True)
                sk_grouped_points = tf.tile(sk_grouped_points, [1, 1, nsample_list[-1], 1])
                grouped_points = tf.concat([sk_grouped_points, grouped_points-sk_grouped_points], axis=-1)

        # local point
        nsample = nsample_list[-1]
        if nsample != 0:
            # grouped_xyz = grouped_xyz_[:,:,int(i*nsample_list[0]):int(i*nsample_list[0])+int(nsample_list[0]),:]
            grouped_xyz1 = grouped_xyz
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
            if points is not None:
                # grouped_points = grouped_points_[:,:,int(i*nsample_list[0]):int(i*nsample_list[0])+int(nsample_list[0]),:]
                if use_xyz:
                    grouped_points = tf.concat([grouped_xyz1, grouped_xyz, grouped_points], axis=-1)
            else:
                grouped_points = tf.concat([grouped_xyz,tf.tile(tf.expand_dims(xyz, 2), [1, 1, nsample, 1])],axis=-1)
        else:
            if points is not None:
                grouped_points = tf.expand_dims(points,axis=-2)
            else:
                grouped_points = tf.expand_dims(xyz,axis=-2)
        c = 0
        # MLP layers
        for i in range(len(nsample_list)):
        	fp = grouped_points[:, :, 0:int(i * grouped_points.get_shape()[2].value // len(nsample_list)) + int(grouped_points.get_shape()[2].value // len(nsample_list)), :]
        	for j,num_out_channel in enumerate(mlp_list):
        	    x = fp
        	    sz = x.get_shape()[3].value // num_groups
        	    conv_side_layers = []
        	    for g in range(num_groups):
        	        conv_side = tf_util.conv2d(x[:, :, :, g * sz:g * sz + sz], num_out_channel//num_groups, [1,1],
        	                                        padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
        	                                        scope='conv%d_%d_%d'%(i,j,g), bn_decay=bn_decay)
        	        conv_side_layers.append(conv_side)
        	    feature = tf.concat(conv_side_layers, axis=-1)
        	    feature = channel_shuffle('channel_shuffle', feature, num_groups)
        	# max pool
        	new_points = tf.reduce_max(feature, axis=[2],keep_dims=True)
        	new_points_list.append(new_points)

        # multi-scale area feature
        feature = tf.concat(new_points_list, axis=-2)

        if output_size > 0:
            xfeature = feature
            sz = xfeature.get_shape()[3].value // num_groups
            xfeature_layers = []
            for s in range(num_groups):
                conv_side = tf_util.conv2d(xfeature[:, :, :, s * sz:s * sz + sz], num_out_channel//num_groups, [1,feature.get_shape()[2].value],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d'%(s), bn_decay=bn_decay)
                xfeature_layers.append(conv_side)
            feature = tf.concat(xfeature_layers, axis=-1)
            feature = channel_shuffle('channel_shuffle', feature, num_groups)

            # feature = tf_util.conv2d(feature, output_size, [1,feature.get_shape()[2].value],
            #                          padding='VALID', stride=[1,feature.get_shape()[2].value], bn=bn, is_training=is_training,
            #                          scope='conv', bn_decay=bn_decay)
            # new_feature = tf.concat(new_feature_list, axis=-2)
            # print(new_feature)
            # exit()
            # new_feature = tf_util.conv2d(new_feature, output_size, [1,feature.get_shape()[2].value],
            #                                 padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
            #                                 scope='conv_f', bn_decay=bn_decay)
            # feature = tf.squeeze(tf.concat([new_feature,feature], axis=-1),axis=2)
        else:
            feature = tf.squeeze(feature)
        return new_xyz, feature

def find_xyz(point_cloud, point_idx):
    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value
    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point = tf.gather(point_cloud_flat, point_idx+idx_)
    return point

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def self_attention(x, mlp, name, is_training, bn_decay):
    with tf.variable_scope(name):
        m_batchsize = x.get_shape()[0].value
        width = x.get_shape()[1].value
        height = x.get_shape()[2].value
        C = x.get_shape()[3].value

        q = tf_util.conv2d(x, mlp, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='query_s', bn_decay=bn_decay)

        k = tf_util.conv2d(x, mlp, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='key_s', bn_decay=bn_decay)

        v = tf_util.conv2d(x, mlp, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='value_s', bn_decay=bn_decay)

        q = tf.reduce_max(q, axis=[1], keep_dims=True, name='maxpool')
        k = tf.reduce_mean(k, axis=[1], keep_dims=True, name='avgpool')

        entropy = tf.matmul(hw_flatten(q), hw_flatten(k), transpose_a=True) # # [bs, N, N]
        attention = tf.nn.softmax(entropy)

        out = tf.matmul(hw_flatten(v),entropy, transpose_b=True)
        out = tf.reshape(out, [m_batchsize,width,height,C])

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        out = gamma*out + x

    return out

def distance_matrix(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keep_dims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keep_dims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    D = tf.sqrt(D)
    return D

def get_glp(point_cloud, sk_point, nn_idx, k, c=False, adj=None, s=False, h=False):
    point_cloud_neighbors = find_xyz(point_cloud, nn_idx)
    if h:
        sk_point,_ = tf.nn.moments(point_cloud_neighbors, axes=2,keep_dims=True)

    else:
        sk_point = tf.expand_dims(sk_point, axis=-2)
    # sk_point = tf.expand_dims(sk_point, axis=-2)
    sk_point = tf.tile(sk_point, [1, 1, k, 1])
    glp_feature = tf.concat([sk_point, point_cloud_neighbors-sk_point], axis=-1)
    return glp_feature

def my_func(dist,idx,randii,nsample):
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            # ind = np.where(dist[i][j]<randii)[0]

            idx[i,j,:] = ind[np.random.choice(ind.shape[0], nsample, replace=False)]
    return idx