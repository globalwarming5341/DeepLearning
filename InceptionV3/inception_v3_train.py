from model.inception_v3 import InceptionV3
import os
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = InceptionV3((128, 128, 3))
    model.train('./data_classification', 64, 0.0001, 20, save_path='inceptionV3.hdf5')