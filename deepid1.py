#! /usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from vec import *

__all__ = ['deepid']
#正态分布
def weight_variable(shape):
    return fluid.layers.create_parameter(shape=shape, dtype='float32',attr=fluid.initializer.Normal(loc=0.0, scale=2.0))
def bias_variable(shape):
    return fluid.layers.zeros(shape=shape, dtype='int64')

#构造卷积池化层
def conv_pool_layer(x, num_filters, w_shape, b_shape,only_conv=False):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv =fluid.layers.conv2d(input=x, num_filters=num_filters, filter_size=W,stride = 1, act="relu")
        # tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        h = conv + b
        relu = fluid.layers.relu(x)
        if only_conv == True:
            return relu
        pool = fluid.layers.pool2d(input=relu,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
        # pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
        return pool
def Wx_plus_b(weights, x, biases):
        return fluid.layers.matmul(x, weights) + biases

def nn_layer(input_tensor, input_dim, output_dim,act=False):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = Wx_plus_b(weights, input_tensor, biases)
        if act != False:
            activations = fluid.layers.relu(preactivate)
            return activations
        else:
            return preactivate

def deepid(input,class_num):
    h1 = conv_pool_layer(input,20, [4, 4, 3], [20])
    h2 = conv_pool_layer(h1,40, [3, 3, 20], [40])
    h3 = conv_pool_layer(h2,60, [3, 3, 40], [60])
    h4 = conv_pool_layer(h3,80, [2, 2, 60], [80], only_conv=True)
    h3r = fluid.layers.reshape(h3, [-1, 5 * 4 * 60], actual_shape=None, act=None, inplace=True, name=None)
    h4r = fluid.layers.reshape(h3, [-1, 4 * 3 * 80], actual_shape=None, act=None, inplace=True, name=None)
    W1 = weight_variable([5 * 4 * 60, 160])
    W2 = weight_variable([4 * 3 * 80, 160])
    b = bias_variable([160])
    h = fluid.layers.matmul(h3r, W1) + fluid.layers.matmul(h4r, W2) + b
    h5 = fluid.layers.relu(h)
    y = nn_layer(h5,160,class_num,act=False)
    return y

    # tf.scalar_summary('loss', loss)

def accuracy(y_estimate, y_real):
    correct_prediction = fluid.layers.equal((fluid.layers.argmax(x=y_estimate,axis=-1),fluid.layers.argmax(x=y_real,axis=-1)))
    accuracy = fluid.layers.reduce_mean(fluid.layers.cast(x=correct_prediction, dtype='float32'))
    return accuracy

def train_step(loss):
    optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
    return optimizer.minimize(loss)


