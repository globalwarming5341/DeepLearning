#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          GAN.py
@Date:          2018/11/12 18:42:05
@Author:        Zhuang ZM
@Description:   A simple program for GAN
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initial random seed
tf.set_random_seed(1)
np.random.seed(1)

batch_size = 64
learning_rate_g = 0.0001
learning_rate_d = 0.0001
input_size = 5
dims = 15
input_data = np.vstack([np.linspace(-1, 1, dims) for _ in range(batch_size)])

# show our beautiful painting range
plt.plot(input_data[0], 2 * np.power(input_data[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(input_data[0], 1 * np.power(input_data[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()



with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None, input_size])
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, dims)

with tf.variable_scope('Discriminator'):
    label = tf.placeholder(tf.float32, [None, dims])
    D_l0 = tf.layers.dense(label, 128, tf.nn.relu)
    D_layer_1 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid)
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, reuse=True)
    D_layer_2 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, reuse=True)

D_loss = -tf.reduce_mean(tf.log(D_layer_1) + tf.log(1 - D_layer_2))
G_loss = tf.reduce_mean(tf.log(1 - D_layer_2))

train_D = tf.train.AdamOptimizer(learning_rate_d).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(learning_rate_g).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()
for step in range(10000):
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
    y = a * np.power(input_data, 2) + (a-1)
    G_noise = np.random.randn(batch_size, input_size)
    G_data, prob, Dl = sess.run([G_out, D_layer_1, D_loss, train_D, train_G],
                                    {G_in: G_noise, label: y})[:3]
                                    
    if step % 100 == 0:
        plt.cla()
        plt.plot(input_data[0], G_data[0], c='#4AD631', lw=3, label='Generated data',)
        plt.plot(input_data[0], 2 * np.power(input_data[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(input_data[0], 1 * np.power(input_data[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=12)
        plt.draw()
        plt.pause(0.05)

plt.show()
sess.close()