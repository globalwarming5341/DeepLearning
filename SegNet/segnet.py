#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:          segnet.py
@Date:          2019/03/05 10:22:49
@Author:        Zhuang ZM
@Description:   Using SegNet for semantic segmentation
'''

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  
from keras.models import Sequential  
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input,Dropout 
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint   
from keras.models import Model
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt  
import cv2
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 1
KEEP_PROB = 0.5
TRAIN_SAMPLE = 3600
VAL_SAMPLE = 400
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 3
DATA_PATH = ""

def data_generator():
    index = 0
    random_num = [x for x in range(1, TRAIN_SAMPLE + 1)]
    random.shuffle(random_num)
    while True:
        npz = np.load(os.path.join(DATA_PATH, 'data_' + str(random_num[index]) + '.npz'))
        image = npz['image'].astype(np.float32) / 255.0
        label = npz['label'].astype(np.float32) / 255.0
        label[label > 0.5] = 1.0
        label[label <= 0.5] = 0.0
        yield (image, label)
        index += 1
        if index >= VAL_SAMPLE // BATCH_SIZE:
            random.shuffle(random_num)
            index = 0

def val_generator():
    index = TRAIN_SAMPLE // BATCH_SIZE
    while True:
        npz = np.load(os.path.join(DATA_PATH, 'data_' + str(index + 1) + '.npz'))
        image = npz['image'].astype(np.float32) / 255.0
        label = npz['label'].astype(np.float32) / 255.0
        label[label > 0.5] = 1.0
        label[label <= 0.5] = 0.0
        yield (image, label)
        index += 1
        if index >= (TRAIN_SAMPLE + VAL_SAMPLE) // BATCH_SIZE:
            index = TRAIN_SAMPLE // BATCH_SIZE
  
def SegNet():  
    model = Sequential()  

    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL),padding='same',activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))  

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  

    model.add(UpSampling2D(size=(2,2)))  

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
 
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
 
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(1, (1, 1), strides=(1, 1), padding='same'))  
    #model.add(Reshape((n_label,IMAGE_HEIGHT*IMAGE_WIDTH)))  
    #model.add(Permute((2,1)))  
    #model.add(Activation('softmax'))  
    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])  
    #model.summary()  
    return model  

def train(): 
    model = SegNet()
    model_checkpoint = ModelCheckpoint('segnet_model.hdf5', monitor='loss', verbose=1, save_best_only=True)   
    H = model.fit_generator(data_generator(),
                            steps_per_epoch = TRAIN_SAMPLE // BATCH_SIZE,
                            validation_data = val_generator(),
                            validation_steps = VAL_SAMPLE // BATCH_SIZE,
                            epochs=EPOCHS,
                            callbacks=[model_checkpoint])  
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("fig_segnet.png")




if __name__=='__main__':  
    train()  
