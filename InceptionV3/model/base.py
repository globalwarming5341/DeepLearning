import os
import numpy as np
from PIL import Image
class BaseModel(object):
    def __init__(self, input):
        self._model = None
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 3:
            self.image_height = input[0]
            self.image_width = input[1]
            self.image_channel = input[2]
        else:
            self.image_height = input.shape[1].value
            self.image_width = input.shape[2].value
            self.image_channel = input.shape[3].value
            
            
    
    def _get_model(self, img_input, include_top, classes):
        raise NotImplementedError("method _get_model() is not implemented!")
    
    def predict_proba(self, image):
        image = image.astype(np.float32)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) / 255.0
        prediction = self._model.predict(image)
        prediction = prediction.reshape(prediction.shape[1])
        return prediction
    
    def predict_from_npy(self, path):
        npy = np.load(path)
        return np.argmax(self.predict_proba(npy))

    def get_metrics(self, test_path):
        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.
        test_files = os.listdir(test_path)
        size = len(test_files)
        for i, test_file in enumerate(test_files):
            prediction = self.predict_from_npy(os.path.join(test_path, test_file))
            if test_file[0] == 'p' and prediction == 0:
                tp += 1
            elif test_file[0] == 'p' and prediction == 1:
                fn += 1
            elif test_file[0] == 'n' and prediction == 0:
                fp += 1
                Image.fromarray(np.load(os.path.join(test_path, test_file))).save('./error/{}.png'.format(test_file))
            elif test_file[0] == 'n' and prediction == 1:
                tn += 1
            print(test_file, prediction, tp, tn, fp, fn)
            # print('{}/{}'.format(i + 1, size))
        return tp, tn, fp, fn