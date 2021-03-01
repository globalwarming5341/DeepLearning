import os
from unet import Unet

EPOCHS = 15
LEARNING_RATE = 0.0005
BATCH_SIZE = 1
DATA_PATH = './data_segmentation'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    unet = Unet(residual=False)
    unet.train(DATA_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS)