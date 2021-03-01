#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          LinearRegression.py
@Date:          2017/07/28 15:19:58
@Author:        Zhuang ZM
@Description:   A simple program to implement linear regression with graph
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

# Generate fake data for training
x = np.random.rand(100).astype(np.float32) 
noise = np.random.normal(0, 0.05, x.size).astype(np.float32)    
y = 2 * x + 0.5 + noise                                
Weights = tf.Variable(tf.constant(0.1))
biases = tf.Variable(tf.constant(0.0))
prediction = Weights * x + biases

loss = tf.reduce_mean(tf.square(prediction - y)) #cost function
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
b = fig.add_subplot(1, 2, 2)  
plt.xlim(0.0, 199.0)               
plt.ylim(0.0, 1.0)                
a.scatter(x, y) 
plt.ion()
plt.show()

step_x = []
loss_y = [] 
lines = []

for step in range(200):
        sess.run(train_step)
        print("Step: " + str(step + 1) + " Weights: " + str(sess.run(Weights)) + " biases: "
              + str(sess.run(biases)) + " loss:" + str(sess.run(loss)))
        if step > 0:
            a.lines.remove(lines[0]) # remove the previous line
        lines = a.plot(x, sess.run(prediction), 'g-', lw = 3)
        step_x.append(step) 
        loss_y.append(sess.run(loss)) 
        b.plot(step_x, loss_y, 'r', lw = 3) 
        plt.pause(0.01)
        
sess.close()