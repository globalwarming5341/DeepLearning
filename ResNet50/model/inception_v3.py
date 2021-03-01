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

class InceptionV3(BaseModel):
    def __init__(self, input, weight_path=None):
        super(InceptionV3, self).__init__(input)
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 3:
            input = layers.Input(input)
        self._model = self._get_model(input)
        if weight_path:
            self._model.load_weights(weight_path)

    @property
    def output(self):
        return self._model.output

    def load_weights(self, path):
        self._model.load_weights(path)

    def conv2d_bn(self,
                  x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1),
                  name=None):
        
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        bn_axis = 3
        x = layers.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = layers.Activation('relu', name=name)(x)
        return x

    def _get_model(self, img_input, include_top=True, classes=2):

        channel_axis = 3


        x = self.conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = self.conv2d_bn(x, 32, 3, 3, padding='valid')
        x = self.conv2d_bn(x, 64, 3, 3)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding='valid')
        x = self.conv2d_bn(x, 192, 3, 3, padding='valid')
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        branch1x1 = self.conv2d_bn(x, 64, 1, 1)
        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, 1, 1)
        branch3x3 = self.conv2d_bn(branch3x3, 320, 3, 3,
                            strides=(2, 2), padding='valid')

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = self.conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool],
            axis=channel_axis,
            name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2],
                axis=channel_axis,
                name='mixed9_' + str(i))

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))
        if include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(1024, activation='relu', name='fcn')(x)
            x = layers.Dense(classes, activation='softmax', name='predictions')(x)


        model = models.Model(img_input, x, name='inception_v3')

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
                if idx >= data_size:
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
        model_checkpoint = ModelCheckpoint(save_path if save_path else 'inception_v3.hdf5', monitor='loss', verbose=1, save_best_only=False)
        scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=0.00001)
        self._model.fit_generator(self.data_generator(train_data_files, 'train', batch_size, self.image_width, self.image_height, self.image_channel),
                            steps_per_epoch=train_data_size // batch_size,
                            validation_data=self.data_generator(val_data_files, 'val', batch_size, self.image_width, self.image_height, self.image_channel),
                            validation_steps=val_data_size // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[model_checkpoint, scheduler])

    