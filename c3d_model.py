#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

NUM_CLASSES = 13
height = 224
width = 224
channels = 3
num_frames = 64
batch_size = 3


def f_c3d(_input_data, _weights, _biases):
    conv1 = tf.nn.conv3d(_input_data, _weights['wc1'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1')
    conv1 = tf.nn.bias_add(conv1, _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='pool1')

    # Convolution Layer
    conv2 = tf.nn.conv3d(pool1, _weights['wc2'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv2')
    conv2 = tf.nn.bias_add(conv2, _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    # pooling layer
    pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='pool2')

    # Convolution Layer
    conv3 = tf.nn.conv3d(pool2, _weights['wc3a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3a')
    conv3 = tf.nn.bias_add(conv3, _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = tf.nn.conv3d(conv3, _weights['wc3b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3b')
    conv3 = tf.nn.bias_add(conv3, _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    # pooling layer
    pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3')

    # Convolution Layer
    conv4 = tf.nn.conv3d(pool3, _weights['wc4a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4a')
    conv4 = tf.nn.bias_add(conv4, _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = tf.nn.conv3d(conv4, _weights['wc4b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4b')
    conv4 = tf.nn.bias_add(conv4, _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    # pooling layer
    pool4 = tf.nn.max_pool3d(conv4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4')

    # Convolution Layer
    conv5 = tf.nn.conv3d(pool4, _weights['wc5a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5a')
    conv5 = tf.nn.bias_add(conv5, _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = tf.nn.conv3d(conv5, _weights['wc5b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5b')
    conv5 = tf.nn.bias_add(conv5, _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')

    # pooling layer
    pool5 = tf.nn.max_pool3d(conv5, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool5')

    return pool5


def t1_c3d(_input_data, _weights, _biases):

    conv6 = tf.nn.conv3d(_input_data, _weights['twc1'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv6a')
    conv6 = tf.nn.bias_add(conv6, _biases['tbc1'])
    conv6 = tf.nn.relu(conv6, 'relu6a')
    conv6 = tf.nn.conv3d(conv6, _weights['twt1'], strides=[1, 1, 1, 1, 1], padding='VALID', name='conv6b')
    conv6 = tf.nn.bias_add(conv6, _biases['tbt1'])
    conv6 = tf.nn.relu(conv6, 'relu6b')
    pool6 = tf.nn.max_pool3d(conv6, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool6')
    return pool6


def t2_c3d(_input_data, _weights, _biases):

    conv7 = tf.nn.conv3d(_input_data, _weights['tw2'], strides=[1, 1, 1, 1, 1], padding='VALID', name='conv7')
    conv7 = tf.nn.bias_add(conv7, _biases['tb2'])
    conv7 = tf.nn.relu(conv7, 'relu7')
    return conv7


def t3_c3d(_input_data, _weights, _biases):

    conv8 = tf.nn.conv3d(_input_data, _weights['tw3'], strides=[1, 1, 1, 1, 1], padding='VALID', name='conv8')
    conv8 = tf.nn.bias_add(conv8, _biases['tb3'])
    conv8 = tf.nn.relu(conv8, 'relu8')
    # conv8 = tf.nn.max_pool3d(conv8, ksize=[1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='VALID', name='pool8')
    return conv8


def fc1(_input_data, fc_weight, fc_biases, o_weight, o_biases, _dropout):
    dense1 = tf.transpose(_input_data, perm=[0, 1, 4, 2, 3])

    dense1 = tf.reshape(dense1, [batch_size, fc_weight['fcw1'].get_shape().as_list()[0]])
    dense1 = tf.matmul(dense1, fc_weight['fcw1']) + fc_biases['fcb1']

    dense1 = tf.nn.relu(dense1, name='fc1')
    dense1 = tf.nn.dropout(dense1, _dropout)

    out = tf.matmul(dense1, o_weight['ow1']) + o_biases['ob1']
    return out


def fc2(_input_data, fc_weight, fc_biases, o_weight, o_biases, _dropout):

    dense2 = tf.transpose(_input_data, perm=[0, 1, 4, 2, 3])

    dense2 = tf.reshape(dense2, [batch_size, fc_weight['fcw2'].get_shape().as_list()[0]])
    dense2 = tf.matmul(dense2, fc_weight['fcw2']) + fc_biases['fcb2']

    dense2 = tf.nn.relu(dense2, name='fc2')
    dense2 = tf.nn.dropout(dense2, _dropout)

    out = tf.matmul(dense2, o_weight['ow2']) + o_biases['ob2']
    return out


def fc3(_input_data, fc_weight, fc_biases, o_weight, o_biases, _dropout):

    dense3 = tf.transpose(_input_data, perm=[0, 1, 4, 2, 3])

    dense3 = tf.reshape(dense3, [batch_size, fc_weight['fcw3'].get_shape().as_list()[0]])
    dense3 = tf.matmul(dense3, fc_weight['fcw2']) + fc_biases['fcb3']

    dense3 = tf.nn.relu(dense3, name='fc3')
    dense3 = tf.nn.dropout(dense3, _dropout)

    out = tf.matmul(dense3, o_weight['ow3']) + o_biases['ob3']
    return out


def inference_c3d(data, _dropout, fw, fb, tw, tb, fcw, fcb, ow, ob):

    data = tf.reshape(data, (batch_size, 8, 8, width, height, channels))
    data = tf.reshape(data, (batch_size*8, 8, width, height, channels))

    pool5 = f_c3d(data, fw, fb)  # batch_size*4, 1, 7, 7, 512
    pool5 = tf.reshape(pool5, (batch_size, 8, 7, 7, 512))

    if num_frames == 32:
        pool6 = t1_c3d(pool5, tw, tb)
        return fc1(pool6, fcw, fcb, ow, ob, _dropout)
    elif num_frames == 64:
        pool6 = t1_c3d(pool5, tw, tb)
        pool7 = t2_c3d(pool6, tw, tb)
        return fc2(pool7, fcw, fcb, ow, ob, _dropout)
    elif num_frames == 128:
        pool6 = t1_c3d(pool5, tw, tb)
        conv7 = t2_c3d(pool6, tw, tb)
        pool7 = tf.nn.max_pool3d(conv7, ksize=[1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='SAME', name='pool7')
        conv8 = t3_c3d(pool7, tw, tb)
        return fc3(conv8, fcw, fcb, ow, ob, _dropout)

