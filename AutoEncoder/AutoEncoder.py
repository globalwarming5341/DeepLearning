#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          AutoEncoder.py
@Date:          2021/02/16 11:25:07
@Author:        Zhuang ZM
@Description:   A simple program for auto encoder
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep

def xavier_init(fan_in, fan_out):
    low = -1 * np.sqrt(6.0 / (fan_in + fan_out))
    high = -low
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)

class AutoEncoder(object):
    def __init__(self, n_inputs, n_hiddens, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.1):
        self.n_input = n_inputs
        self.n_hidden = n_hiddens
        self.scale = scale
        self.weights = {
            'weight1': tf.Variable(xavier_init(self.n_input, self.n_hidden)),
            'weight2': tf.Variable(tf.zeros([self.n_hidden, self.n_input]))
        }
        self.biases = {
            'bias1': tf.zeros([self.n_hidden]),
            'bias2': tf.zeros([self.n_input])
        }
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = tf.matmul(self.x + scale * tf.random_normal((n_inputs, )),
                                self.weights['weight1']) + self.biases['bias1']
        self.reconstruction = tf.matmul(self.hidden, self.weights['weight2']) + self.biases['bias2']
        self.loss = 0.5 * tf.reduce_sum(tf.square(self.reconstruction - self.x))
        self.optimizer = optimizer.minimize(self.loss)
 
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def partial_fit(self, X):
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict={self.x: X})
        return loss

    def calc_cost(self, X):
        return self.sess.run(self.loss, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

# input mnist data with one hot label
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
training_epoch = 20
batch_size = 64

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: start_index + batch_size]

# normalize input data using StandardScaler
preprocessor = prep.StandardScaler().fit(mnist.train.images)
X_train = preprocessor.transform(mnist.train.images)
X_test = preprocessor.transform(mnist.test.images)

autoencoder = AutoEncoder(784, 200)

for epoch in range(training_epoch):
    mean_loss = 0
    step = 1
    while step * batch_size < n_samples:
        idx = np.random.randint(0, len(X_train) - batch_size)
        batch_x = X_train[idx: idx + batch_size]
        loss = autoencoder.partial_fit(batch_x)
        mean_loss += loss / batch_size
        step += 1
    print('Epoch: %d' % epoch, ', Loss: %.2f' % mean_loss)


encoder_test = autoencoder.reconstruct(X_test[:10])

fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(len(encoder_test)):
    ax[0][i].imshow(X_test[i].reshape((28, 28)), cmap='Greys', interpolation='nearest')
    ax[1][i].imshow(encoder_test[i].reshape((28, 28)), cmap='Greys', interpolation='nearest')
plt.tight_layout()
plt.show()