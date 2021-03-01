from model.resnet50 import ResNet50
import os
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = ResNet50((128, 128, 3))
    model.train('./data_classification', 64, 0.0001, 20, save_path='resnet.hdf5')