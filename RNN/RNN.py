#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          RNN.py
@Date:          2017/07/30 21:04:25
@Author:        Zhuang ZM
@Description:   A simple program for RNN
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input mnist data with one hot label
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

iteration_num = 10000
batch_size = 32
n_inputs = 28
n_steps = 28
n_hidden_number = 64
n_outputs = 10

x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
Y = tf.placeholder(tf.float32,[None,n_outputs])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_number])),
    'out': tf.Variable(tf.random_normal([n_hidden_number,n_outputs]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape = [n_hidden_number,])),
    'out': tf.Variable(tf.constant(0.1, shape = [n_outputs,]))
}


def RNN(X,weights,biases):
    X = tf.reshape(X,[-1, n_inputs]) # flatten X in order to multiply

    inputs = tf.matmul(X, weights['in']) + biases['in']

    inputs = tf.reshape(inputs, [-1, n_steps, n_hidden_number])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_number, forget_bias = 1.0)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    
    outputs,states = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=init_state, time_major=False)

    return tf.matmul(states[1], weights['out']) + biases['out']

prediction = RNN(x, weights, biases)

# cross entropy loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
step = 0

while step * batch_size < iteration_num:
    batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs]) # [100, 28, 28]
    train_step.run(feed_dict={x: batch_xs,Y: batch_ys,})
    if step % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs,Y: batch_ys,})
        print("Step : ", step, ", Acc :", train_accuracy)
    step += 1

sess.close()