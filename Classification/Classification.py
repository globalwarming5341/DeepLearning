#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          Classification.py
@Date:          2017/07/29 12:28:45
@Author:        Zhuang ZM
@Description:   A simple program to implement classification with graph
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# initial random seed
tf.set_random_seed(1)
np.random.seed(1)

# generate some fake data for training
original_data = np.ones((100, 2))
x = np.vstack((np.random.normal(2 * original_data, 1), np.random.normal(-2 * original_data, 1)))
y = np.hstack((np.zeros(100), np.ones(100)))

placeholder_x = tf.placeholder(tf.float32, x.shape)
placeholder_y = tf.placeholder(tf.int32, y.shape)

layer1 = tf.layers.dense(placeholder_x, 10, tf.nn.relu)
output = tf.layers.dense(layer1, 2)

loss = tf.losses.sparse_softmax_cross_entropy(labels=placeholder_y, logits=output) 
# placeholder_y looks like [0, 1, 2, 1, 2] 

# loss = tf.losses.softmax_cross_entropy(onehot_labels=placeholder_y, logits=output)
# placeholder_y looks like [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]

acc = tf.metrics.accuracy(labels=tf.squeeze(placeholder_y), predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

plt.ion()
for step in range(100):
    _, acc, pred = sess.run([train, acc, output], {placeholder_x: x, placeholder_y: y})
    if step % 2 == 0:
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'acc = %.2f' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.15)

plt.show()