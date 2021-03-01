from model.inception_v3 import InceptionV3
import os
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = InceptionV3((128, 128, 3), 'i05.hdf5')
    test_path = './data_classification_test'
    print(model.get_metrics(test_path))    

    