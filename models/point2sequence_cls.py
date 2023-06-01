'''
Basic Point2Sequence classification model
'''
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import *
import math
import pointfly as pf

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl



def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification Point2Sequence, input is BxNx3, output BxCLASSES """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    alpha = tf.get_variable("alpha", [1], initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable("beta", [1], initializer=tf.constant_initializer(1.0))
    l0_xyz = point_cloud

    l0_xyz, l0_points = point2sequence_module(l0_xyz, None, None, None, [20], [64], 0, is_training, bn_decay, scope='layer0',radi=[0.2], select=False)

    l1_xyz, l1_points = point2sequence_module(l0_xyz, l0_points, 512, None, [16,32,48,64], [128], 128, is_training, bn_decay, scope='layer1',radi=[0.28])

    l2_xyz, l2_points = point2sequence_module(l1_xyz, l1_points, 128, None, [16,32], [256], 256, is_training, bn_decay, scope='layer2',radi=[0.35])

    # l3_xyz, l3_points = point2sequence_module(l2_xyz, l2_points, 64, 0, [16], [512], 0, 0, is_training, bn_decay, scope='layer3',radi=[0.42])

    # l4_points = tf_util.conv1d(l3_points, 1024, l3_points.get_shape()[1].value, padding='VALID', stride=1, bn=True, is_training=is_training, scope='layer4', bn_decay=bn_decay)

    l4_points = conv(l2_points, kernel=[1,1], mlp = [256], stride=[1,1], is_training=is_training, scope='layer4', bn_decay=bn_decay,max_pooling=True)

    # feature = tf.concat([l1_points,l2_points,l3_points],axis=1)
    # l4_points = conv(l1_points, kernel=[32,1], mlp = [512], stride=[8,1], is_training=is_training, scope='layer4', bn_decay=bn_decay)

    # l4_points = conv(l2_points, kernel=[16,1], mlp = [512], stride=[4,1], is_training=is_training, scope='layer4', bn_decay=bn_decay)
    # l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, tf.squeeze(l2_points, [2]), npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Fully connected layers
    net = tf.reshape(l4_points, [batch_size, -1])
    # net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, l1_xyz, l2_xyz, l2_xyz


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
