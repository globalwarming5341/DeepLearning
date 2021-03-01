import os
import numpy as np
import random
import cv2
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from .base import BaseModel

class ResNet50(BaseModel):
    def __init__(self, input):
        super(ResNet50, self).__init__(input)
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 3:
            input = layers.Input(input)
        self._model = self._get_model(input)

    @property
    def output(self):
        return self._model.output

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        bn_axis = 3
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        x = layers.Conv2D(filters1, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                        padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(self, 
                    input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    strides=(2, 2)):

        bn_axis = 3
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1), strides=strides,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                kernel_initializer='he_normal',
                                name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def _get_model(self, img_input, include_top=True, classes=2):

        bn_axis = 3
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer='he_normal',
                        name='conv1')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        if include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(1024, activation='relu', name='fcn')(x)
            x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    

        model = models.Model(img_input, x, name='resnet50')

        return model
    
    def data_generator(self, data_files, data_type, batch_size, width, height, channel):
        data_size = len(data_files)
        random_index = np.random.permutation(data_size)
        idx = 0
        while True:
            image = np.zeros(shape=(batch_size, height, width, channel), dtype=np.float32)
            label = np.zeros(shape=(batch_size, 2), dtype=np.uint8)
            i = 0
            while i < batch_size:
                image[i] = np.load(os.path.join(self._data_path, data_type, data_files[random_index[idx]]))[:, :, :].astype(np.float32)
                if data_files[random_index[idx]].startswith('p'):
                    label[i, 0] = 1
                else:
                    label[i, 1] = 1
                i += 1
                idx += 1
                if  idx >= data_size:
                    idx = 0
                    if data_type == 'train':
                        random_index = np.random.permutation(data_size)
                    
            image = image / 255.0
            yield image, label
    
    def train(self, data_path, batch_size, learning_rate, epochs, save_path=None):
        self._data_path = data_path
        train_data_files = os.listdir(os.path.join(data_path, 'train'))
        train_data_size = len(train_data_files)
        train_data_files = train_data_files[:train_data_size // batch_size * batch_size]


        val_data_files = os.listdir(os.path.join(data_path, 'val'))
        val_data_size = len(val_data_files)
        val_data_files = val_data_files[:val_data_size // batch_size * batch_size]
        
        
        self._model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        model_checkpoint = ModelCheckpoint(save_path if save_path else 'resnet.hdf5', monitor='loss', verbose=1, save_best_only=False)
        scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=0.00001)
        self._model.fit_generator(self.data_generator(train_data_files, 'train', batch_size, self.image_width, self.image_height, self.image_channel),
                            steps_per_epoch=train_data_size // batch_size,
                            validation_data=self.data_generator(val_data_files, 'val', batch_size, self.image_width, self.image_height, self.image_channel),
                            validation_steps=val_data_size // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[model_checkpoint, scheduler])