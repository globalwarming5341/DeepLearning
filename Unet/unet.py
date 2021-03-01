# -*- coding=utf-8 -*-
#Copyright: Shenzhen University
#File: unet.py
#Author: Zhuang Zemin
#Date: 2019/10/1
#Description: U-net Model
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, PReLU, concatenate, BatchNormalization, Add
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K

def binary_crossentropy(y_true, y_pred):
    mask = tf.where(tf.equal(y_true, 0), np.zeros((1, 512, 512, 1), dtype=np.float32),  np.ones((1, 512, 512, 1), dtype=np.float32))
    y_true = tf.where(tf.equal(y_true, 2),  np.ones((1, 512, 512, 1), dtype=np.float32),  np.zeros((1, 512, 512, 1), dtype=np.float32))
    crossentropy = K.binary_crossentropy(y_true, y_pred) * mask
    return K.mean(crossentropy, axis=-1)

def dice_coef(y_true, y_pred, smooth=1):
    mask = tf.where(tf.equal(y_true, 0), np.zeros((1, 512, 512, 1), dtype=np.float32),  np.ones((1, 512, 512, 1), dtype=np.float32))
    y_true = tf.where(tf.equal(y_true, 2),  np.ones((1, 512, 512, 1), dtype=np.float32),  np.zeros((1, 512, 512, 1), dtype=np.float32))
    intersection = K.sum(y_true * y_pred * mask)
    union = K.sum(y_true * mask) + K.sum(y_pred * mask)
    dice_coef = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice_coef)

def total_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1. - dice_coef(y_true, y_pred))


class Unet(object):
    def __init__(self, image_width=512, image_height=512, image_channel=3, weight_path=None, residual=False):
        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel
        if residual:
            self._model = self.get_resunet()
        else:
            self._model = self.get_unet()
        if weight_path:
            self._model.load_weights(weight_path)
        self._data_path = ''
        self._data_size = 0

    def convBlock(self, inputs, filters):
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        return x
    
    def upBlock(self, inputs, filters):
        x = UpSampling2D(size=(2, 2))(inputs)
        x = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal')(x)
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


    def get_resunet(self):
        inputs = Input((self.image_height, self.image_width, self.image_channel))

        # conv1 = self.residual_block(inputs, 16)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # conv2 = self.residual_block(pool1, 32)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # conv3 = self.residual_block(pool2, 64)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = self.residual_block(pool3, 128)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # conv5 = self.residual_block(pool4, 256)
        # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # conv6 = self.residual_block(pool5, 512)
        # pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

        # conv7 = self.residual_block(pool6, 1024)
        # pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

        # conv8 = self.residual_block(pool7, 512)
        # up1 = UpSampling2D(size=(2, 2))(conv8)
        # concat1 = concatenate([conv7, up1])

        # conv9 = self.residual_block(concat1, 256)
        # up2 = UpSampling2D(size=(2, 2))(conv9)
        # concat2 = concatenate([conv6, up2])

        # conv10 = self.residual_block(concat2, 128)
        # up3 = UpSampling2D(size=(2, 2))(conv10)
        # concat3 = concatenate([conv5, up3])

        # conv11 = self.residual_block(concat3, 64)
        # up4 = UpSampling2D(size=(2, 2))(conv11)
        # concat4 = concatenate([conv4, up4])

        # conv12 = self.residual_block(concat4, 32)
        # up5 = UpSampling2D(size=(2, 2))(conv12)
        # concat5 = concatenate([conv3, up5])

        # conv13 = self.residual_block(concat5, 16)
        # up6 = UpSampling2D(size=(2, 2))(conv13)
        # concat6 = concatenate([conv2, up6])

        # conv14 = self.residual_block(concat6, 8)
        # up7 = UpSampling2D(size=(2, 2))(conv14)
        # concat7 = concatenate([conv1, up7])

        # outputs = Conv2D(1, 1, activation='sigmoid')(concat7)

        # model = Model(inputs=inputs, outputs=outputs, name='resunet')
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
        drop6 = Dropout(0.5)(conv6)
        pool6 = MaxPooling2D(pool_size=(2, 2))(drop6)

        conv7 = self.residual_block(pool6, 1024)
        drop7 = Dropout(0.5)(conv7)

        up8 = self.upBlock(drop7, 512)
        concat8 = concatenate([drop6, up8])
        conv8 = self.residual_block(concat8, 512)

        up9 = self.upBlock(conv8, 256)
        concat9 = concatenate([conv5, up9])
        conv9 = self.residual_block(concat9, 256)

        up10 = self.upBlock(conv9, 128)
        concat10 = concatenate([conv4, up10])
        conv10 = self.residual_block(concat10, 128)

        up11 = self.upBlock(conv10, 64)
        concat11 = concatenate([conv3, up11])
        conv11 = self.residual_block(concat11, 64)

        up12 = self.upBlock(conv11, 32)
        concat12 = concatenate([conv2, up12])
        conv12 = self.residual_block(concat12, 32)

        up13 = self.upBlock(conv12, 16)
        concat13 = concatenate([conv1, up13])
        conv13 = self.residual_block(concat13, 16)


        conv13 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv13)
        conv13 = BatchNormalization()(conv13)
        conv13 = PReLU()(conv13)
        conv14 = Conv2D(1, 1, activation = 'sigmoid')(conv13)

        model = Model(inputs=inputs, outputs=conv14, name='resunet')

        return model

    def get_unet(self):
        inputs = Input((self.image_height, self.image_width, self.image_channel))
      
        conv1 = self.convBlock(inputs, 64)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.convBlock(pool1, 128)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.convBlock(pool2, 256)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.convBlock(pool3, 512)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = self.convBlock(pool4, 1024)
        drop5 = Dropout(0.5)(conv5)

        up6 = self.upBlock(drop5, 512)
        concat6 = concatenate([drop4, up6])
        conv6 = self.convBlock(concat6, 512)

        up7 = self.upBlock(conv6, 256)
        concat7 = concatenate([conv3, up7])
        conv7 = self.convBlock(concat7, 256)

        up8 = self.upBlock(conv7, 128)
        concat8 = concatenate([conv2, up8])
        conv8 = self.convBlock(concat8, 128)

        up9 = self.upBlock(conv8, 64)
        concat9 = concatenate([conv1, up9])
        conv9 = self.convBlock(concat9, 64)
        conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = PReLU()(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10, name='unet')
        

        
        # # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # conv1 = self.convBlock(inputs, 16)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        # # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # conv2 = self.convBlock(pool1, 32)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


        # # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        # # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # conv3 = self.convBlock(pool2, 64)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        # # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

        # conv4 = self.convBlock(pool3, 128)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        # # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

        # conv5 = self.convBlock(pool4, 256)
        # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # conv6 = self.convBlock(pool5, 512)
        # pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

        # conv7 = self.convBlock(pool6, 1024)
        # pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

        # conv8 = self.convBlock(pool7, 512)
        # up1 = UpSampling2D(size=(2, 2))(conv8)
        # concat1 = concatenate([conv7, up1])

        # conv9 = self.convBlock(concat1, 256)
        # up2 = UpSampling2D(size=(2, 2))(conv9)
        # concat2 = concatenate([conv6, up2])

        # conv10 = self.convBlock(concat2, 128)
        # up3 = UpSampling2D(size=(2, 2))(conv10)
        # concat3 = concatenate([conv5, up3])

        # conv11 = self.convBlock(concat3, 64)
        # up4 = UpSampling2D(size=(2, 2))(conv11)
        # concat4 = concatenate([conv4, up4])

        # conv12 = self.convBlock(concat4, 32)
        # up5 = UpSampling2D(size=(2, 2))(conv12)
        # concat5 = concatenate([conv3, up5])

        # conv13 = self.convBlock(concat5, 16)
        # up6 = UpSampling2D(size=(2, 2))(conv13)
        # concat6 = concatenate([conv2, up6])

        # conv14 = self.convBlock(concat6, 8)
        # up7 = UpSampling2D(size=(2, 2))(conv14)
        # concat7 = concatenate([conv1, up7])

        # outputs = Conv2D(1, 1, activation='sigmoid')(concat7)

        # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv5))
        # merge6 = concatenate([drop4, up6])
        # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        #
        # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv6))
        # merge7 = concatenate([conv3, up7])
        # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        #
        # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv7))
        # merge8 = concatenate([conv2, up8])
        # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        #
        # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv8))
        # merge9 = concatenate([conv1, up9])
        # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        # model = Model(inputs=inputs, outputs=outputs, name='unet')
        return model

    def data_generator(self, data_files, data_type, width, height, channel):
        data_size = len(data_files)
        random_index = np.random.permutation(data_size)
        idx = 0
        while True:
            image = np.zeros(shape=(self._batch_size, height, width, channel), dtype=np.float32)
            label = np.zeros(shape=(self._batch_size, height, width, 1), dtype=np.uint8)
            i = 0
            while i < self._batch_size:
                image[i] = np.load(os.path.join(self._data_path, data_type, data_files[random_index[idx]]))[:, :, :3].astype(np.float32)
                label[i] = np.load(os.path.join(self._data_path, data_type, data_files[random_index[idx]]))[:, :, [3]].astype(np.uint8)
                i += 1
                idx += 1
                if idx >= data_size:
                    idx = 0
                    if data_type == 'train':
                        random_index = np.random.permutation(data_size)
            image = image / 255.0
            yield image, label

    # def val_data_generator(self, data_files, batch_size, width, height, channel):
    #     data_size = len(data_files)
    #     random_index = np.random.permutation(data_size)
    #     idx = 0
    #     while True:
    #         image = np.zeros(shape=(batch_size, height, width, channel), dtype=np.float32)
    #         label = np.zeros(shape=(batch_size, height, width, 1), dtype=np.uint8)
    #         i = 0
    #         while i < batch_size:
    #             image[i] = np.load(os.path.join(self._data_path, 'val', data_files[random_index[idx]]))[:, :, :3].astype(np.float32)
    #             label[i] = np.load(os.path.join(self._data_path, 'val', data_files[random_index[idx]]))[:, :, [3]].astype(np.uint8)
    #             i += 1
    #             idx += 1
    #             if idx >= data_size:
    #                 random_index = np.random.permutation(data_size)
    #                 idx = 0
    #         image = image / 255.0
    #         yield image, label
    
    def load_data(self, data_path):
        data_files = os.listdir(data_path)
        data_size = len(data_files)
        data_files = data_files[:data_size // self._batch_size * self._batch_size]
        return data_files, data_size

    def train(self, data_path, batch_size, learning_rate, epochs):
        self._data_path = data_path
        self._batch_size = batch_size
        
        train_data_files, train_data_size = self.load_data(os.path.join(data_path, 'train'))
        val_data_files, val_data_size = self.load_data(os.path.join(data_path, 'val'))

        
        self._model.compile(optimizer=Adam(lr=learning_rate), loss=total_loss, metrics=[dice_coef])
        model_checkpoint = ModelCheckpoint('{}.hdf5'.format(self._model.name), monitor='loss', verbose=1, save_best_only=False)
        scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, min_lr=0.00001)
        self._model.fit_generator(self.data_generator(train_data_files, 'train', self.image_width, self.image_height, self.image_channel),
                            steps_per_epoch=train_data_size // batch_size,
                            validation_data=self.data_generator(val_data_files, 'val', self.image_width, self.image_height, self.image_channel),
                            validation_steps=val_data_size // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[model_checkpoint, scheduler])

    def predict_proba(self, image):
        image = image.astype(np.float32)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) / 255.0
        prediction = self._model.predict(image).reshape((image.shape[1], image.shape[2]))
        return prediction
    
    def predict_from_image(self, image_path, iter_num=0):
        original = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        prediction_count = np.zeros(shape=(original.shape[0], original.shape[1]), dtype=np.uint8)
        result = np.zeros(shape=(original.shape[0], original.shape[1]), dtype=np.float32)
        x = 0
        y = 0
        x_strides = self.image_width // 8
        y_strides = self.image_height // 8
        while True:
            crop_image = original[y: y + self.image_height, x: x + self.image_width, :]
            prediction = self.predict_proba(crop_image)
            result[y: y + self.image_height, x: x + self.image_width] += prediction
            prediction_count[y: y + self.image_height, x: x + self.image_width] += 1

            x += x_strides
            if x + self.image_width > original.shape[1]:
                x = 0
                y += y_strides
                if y + self.image_height > original.shape[0]:
                    break
        if iter_num:
            for i in range(iter_num):
                x = random.randint(0, original.shape[1] - self.image_width)
                y = random.randint(0, original.shape[0] - self.image_height)
                crop_image = original[y: y + self.image_height, x: x + self.image_width, :]
                prediction = self.predict_proba(crop_image)
                result[y: y + self.image_height, x: x + self.image_width] += prediction
                prediction_count[y: y + self.image_height, x: x + self.image_width] += 1
        result = result / prediction_count
        return result








