'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import modelnet_dataset
import modelnet_h5_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  #set your gpu id

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='point2sequence_cls', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 
# SHAPE_NAMES = [line.rstrip() for line in \
#     open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))] 
HOSTNAME = socket.gethostname()

# Shapenet official train/test split
if FLAGS.normal:
    assert(NUM_POINT <= 10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, modelnet10=False, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, modelnet10=False, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT <= 2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)
# DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048')
# TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, modelnet10=False, batch_size=BATCH_SIZE)
# TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, modelnet10=False, batch_size=BATCH_SIZE)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points, xyz1, xyz2, xyz3 = MODEL.get_model(pointclouds_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'xyz1':xyz1,
           'xyz2':xyz2,
           'xyz3':xyz3,
           'loss': total_loss}

    eval_one_epoch(sess, ops, num_votes)

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            loss_val, pred_val, xyz1, xyz2, xyz3 = sess.run([ops['loss'], ops['pred'], ops['xyz1'], ops['xyz2'], ops['xyz3']], feed_dict=feed_dict)
            
            # x1=[k[0] for k in cur_batch_data[10]]
            # y1=[k[1] for k in cur_batch_data[10]]
            # z1=[k[2] for k in cur_batch_data[10]]
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.scatter(x1, y1, z1,c='r',marker='o',s=2)
            # # ax.scatter(x,y,z,c='b',marker='o',s=5,)
            # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'r'})
            # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'r'})
            # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'r'})
            # plt.show()
            # print(cur_batch_label[10])

            x1=[k[0] for k in xyz1[10]]
            y1=[k[1] for k in xyz1[10]]
            z1=[k[2] for k in xyz1[10]]
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x1, y1, z1,c='r',marker='o',s=2)
            # ax.scatter(x,y,z,c='b',marker='o',s=5,)
            ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'r'})
            ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'r'})
            ax.set_xlabel('X', fontdict={'size': 15, 'color': 'r'})
            plt.show()

            x1=[k[0] for k in xyz2[10]]
            y1=[k[1] for k in xyz2[10]]
            z1=[k[2] for k in xyz2[10]]
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x1, y1, z1,c='r',marker='o',s=8)
            # ax.scatter(x,y,z,c='b',marker='o',s=5,)
            ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'r'})
            ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'r'})
            ax.set_xlabel('X', fontdict={'size': 15, 'color': 'r'})
            plt.show()

            x1=[k[0] for k in xyz3[10]]
            y1=[k[1] for k in xyz3[10]]
            z1=[k[2] for k in xyz3[10]]
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x1, y1, z1,c='r',marker='o',s=14)
            # ax.scatter(x,y,z,c='b',marker='o',s=5,)
            ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'r'})
            ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'r'})
            ax.set_xlabel('X', fontdict={'size': 15, 'color': 'r'})
            plt.show()
            exit()
            batch_pred_sum += pred_val
        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
