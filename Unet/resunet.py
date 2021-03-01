#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          resunet.py
@Date:          2017/11/24 20:49:31
@Author:        Zhuang ZM
@Description:   
'''

import numpy as np
import glob
import matplotlib.pyplot as plt
import random
import os
import cv2
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, PReLU, concatenate, BatchNormalization, Add
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

import matplotlib
matplotlib.use('Agg')

EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
KEEP_PROB = 0.4
TRAIN_SAMPLE = 180000
VAL_SAMPLE = 20000
DATA_PATH = './datasets'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

height = 512
width = 512
channel = 3
batch_size = 16
epochs = 20
cancer_path = './cancer'
non_cancer_path = './non_cancer'

train_sample_ratio = 0.9

cancer_files = np.array(glob.glob(os.path.join(cancer_path, 'image', '*.png')))
non_cancer_files = np.array(glob.glob(os.path.join(non_cancer_path, '*.png')))
cancer_random_index = np.random.permutation(len(cancer_files))
non_cancer_random_index = np.random.permutation(len(non_cancer_files))

train_cancer_files = cancer_files[cancer_random_index[:int(train_sample_ratio * len(cancer_files))]]
train_non_cancer_files = non_cancer_files[non_cancer_random_index[:int(train_sample_ratio * len(non_cancer_files))]]
val_cancer_files = cancer_files[cancer_random_index[int(train_sample_ratio * len(cancer_files)):]]
val_non_cancer_files = non_cancer_files[non_cancer_random_index[int(train_sample_ratio * len(non_cancer_files)):]]

train_cancer_files_max = len(train_cancer_files) // batch_size * batch_size
train_cancer_files = train_cancer_files[:train_cancer_files_max]
train_non_cancer_files_max = len(train_non_cancer_files) // batch_size * batch_size
train_non_cancer_files = train_non_cancer_files[:train_non_cancer_files_max]

nb_train_samples = train_cancer_files_max + train_non_cancer_files_max
nb_val_samples = len(val_cancer_files) + len(val_non_cancer_files)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def dice_coef_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return 1 - K.mean((2. * intersection + smooth) / (union + smooth))

def total_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def train_data_generator(batch_size=16):
    cancer_random_index = np.random.permutation(len(train_cancer_files))
    non_cancer_random_index = np.random.permutation(len(train_non_cancer_files))
    cancer_index = 0
    non_cancer_index = 0
    batch_size_half = batch_size // 2
    while True:
        batch_random_index = np.random.permutation(batch_size)
        image = np.zeros(shape=(batch_size, height, width, channel), dtype=np.float32)
        label = np.zeros(shape=(batch_size, height, width, 1), dtype=np.float32)
        i = 0
        while i < batch_size:
            if i < batch_size_half:
                try:
                    loaded_img = cv2.imread(train_cancer_files[cancer_random_index[cancer_index]], cv2.IMREAD_COLOR)
                    filename = os.path.basename(train_cancer_files[cancer_random_index[cancer_index]])
                    loaded_label = cv2.imread(os.path.join(cancer_path, 'label', filename), cv2.IMREAD_GRAYSCALE).reshape(height, width, 1).astype(np.float32)
                    image[batch_random_index[i]] = cv2.cvtColor(
                        loaded_img,
                        cv2.COLOR_BGR2RGB).astype(np.float32)
                    label[batch_random_index[i]] = loaded_label
                    cancer_index += 1

                    if cancer_index >= train_cancer_files_max:
                        cancer_index = 0
                        cancer_random_index = np.random.permutation(len(train_cancer_files))
                    i += 1
                except IndexError as e:
                    #print(e)
                    cancer_index = 0
                    cancer_random_index = np.random.permutation(len(train_cancer_files))
            else:
                try:
                    loaded_img = cv2.imread(train_non_cancer_files[non_cancer_random_index[non_cancer_index]], cv2.IMREAD_COLOR)
                    loaded_label = np.zeros(shape=(height, width, 1), dtype=np.float32)
                    image[batch_random_index[i]] = cv2.cvtColor(
                        loaded_img,
                        cv2.COLOR_BGR2RGB).astype(np.float32)
                    label[batch_random_index[i]] = loaded_label
                    non_cancer_index += 1
                    if non_cancer_index >= train_non_cancer_files_max:
                        non_cancer_index = 0
                        non_cancer_random_index = np.random.permutation(len(train_non_cancer_files))
                    i += 1
                except IndexError as e:
                    #print(e)
                    non_cancer_index = 0
                    non_cancer_random_index = np.random.permutation(len(train_non_cancer_files))


        image = image / 255.0
        label = label / 255.0
        yield image, label

def val_data_generator(batch_size=16):
    cancer_index = 0
    non_cancer_index = 0
    batch_size_half = batch_size // 2
    while True:
        image = np.zeros(shape=(batch_size, height, width, channel), dtype=np.float32)
        label = np.zeros(shape=(batch_size, height, width, 1), dtype=np.float32)
        i = 0
        while i < batch_size:
            if i < batch_size_half:
                try:
                    loaded_img = cv2.imread(val_cancer_files[cancer_index], cv2.IMREAD_COLOR)
                    filename = os.path.basename(train_cancer_files[cancer_random_index[cancer_index]])
                    loaded_label = cv2.imread(os.path.join(cancer_path, 'label', filename),
                                              cv2.IMREAD_GRAYSCALE).reshape(height, width, 1).astype(np.float32)
                    image[i] = cv2.cvtColor(
                        loaded_img,
                        cv2.COLOR_BGR2RGB).astype(np.float32)

                    label[i] = loaded_label
                    cancer_index += 1
                    if cancer_index >= train_cancer_files_max:
                        cancer_index = 0
                    i += 1
                except IndexError:
                    cancer_index = 0
            else:
                try:
                    loaded_img = cv2.imread(val_non_cancer_files[non_cancer_index], cv2.IMREAD_COLOR)
                    loaded_label = np.zeros(shape=(height, width, 1), dtype=np.float32)
                    image[i] = cv2.cvtColor(
                        loaded_img,
                        cv2.COLOR_BGR2RGB).astype(np.float32)
                    label[i] = loaded_label
                    non_cancer_index += 1
                    if non_cancer_index >= train_non_cancer_files_max:
                        non_cancer_index = 0
                    i += 1
                except IndexError:
                    non_cancer_index = 0

        image = image / 255.0
        label = label / 255.0
        yield image, label

class Unet(object):
    def __init__(self, image_width=512, image_height=512):
        self.image_width = image_width
        self.image_height = image_height

    def convBlock(self, inputs, filters):
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        return x

    def residual_block(self, inputs, filters):
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        shortcut = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal')(inputs)
        return PReLU()(Add()([x, shortcut]))


    def get_unet(self):
        inputs = Input((self.image_height, self.image_width, 3))

        conv1 = self.residual_block(inputs, 16)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.residual_block(pool1, 32)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.residual_block(pool2, 64)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.residual_block(pool3, 128)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.residual_block(pool4, 256)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        conv6 = self.residual_block(pool5, 512)
        pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

        conv7 = self.residual_block(pool6, 1024)
        pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

        conv8 = self.residual_block(pool7, 512)
        up1 = UpSampling2D(size=(2, 2))(conv8)
        concat1 = concatenate([conv7, up1])

        conv9 = self.residual_block(concat1, 256)
        up2 = UpSampling2D(size=(2, 2))(conv9)
        concat2 = concatenate([conv6, up2])

        conv10 = self.residual_block(concat2, 128)
        up3 = UpSampling2D(size=(2, 2))(conv10)
        concat3 = concatenate([conv5, up3])

        conv11 = self.residual_block(concat3, 64)
        up4 = UpSampling2D(size=(2, 2))(conv11)
        concat4 = concatenate([conv4, up4])

        conv12 = self.residual_block(concat4, 32)
        up5 = UpSampling2D(size=(2, 2))(conv12)
        concat5 = concatenate([conv3, up5])

        conv13 = self.residual_block(concat5, 16)
        up6 = UpSampling2D(size=(2, 2))(conv13)
        concat6 = concatenate([conv2, up6])

        conv14 = self.residual_block(concat6, 8)
        up7 = UpSampling2D(size=(2, 2))(conv14)
        concat7 = concatenate([conv1, up7])

        output = Conv2D(1, 1, activation='sigmoid')(concat7)

        model = Model(input=inputs, output=output)

        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=total_loss, metrics=['accuracy'])

        return model

    def train(self):
        model = self.get_unet()
        model_checkpoint = ModelCheckpoint('resunet.hdf5', monitor='loss', verbose=1, save_best_only=False)
        model.fit_generator(train_data_generator(batch_size),
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=epochs,
                            validation_data=val_data_generator(batch_size),
                            validation_steps=nb_val_samples // batch_size,
                            verbose=1,
                            callbacks=[model_checkpoint])

if __name__ == '__main__':
    unet = Unet()
    unet.train()







