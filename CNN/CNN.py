#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          CNN.py
@Date:          2017/07/29 00:43:27
@Author:        Zhuang ZM
@Description:   A simple program for CNN training
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input mnist data with one hot label
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(x, y, prediction):
    y_pred = sess.run(prediction, feed_dict={xs: x, keep_prob: 1})
    correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return sess.run(acc, feed_dict={xs: x, ys: y, keep_prob: 1})

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    # strides = [1, x_axis_strides, y_axis_strides, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # strides = [1, x_axis_strides, y_axis_strides, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 784]) # image size = 28 x 28 = 784
ys = tf.placeholder(tf.float32, [None, 10]) # 10 classes
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size = 28 x 28 x 32
h_pool1 = max_pool_2x2(h_conv1) # output size = 14 x 14 x 32

# convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size = 14 x 14 x 64
h_pool2 = max_pool_2x2(h_conv2) # output size = 7 x 7 x 64

# fully connected layer 1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # shape = [number of samples, 7 * 7 * 64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fully connected layer 2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cross entropy loss
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 100 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels, prediction))

sess.close()