import shutil
import os
import cv2
import numpy as np



def test_image_split(data_path):
    if not os.path.exists('test'):
        os.mkdir('test')
    if not os.path.exists(os.path.join('test', 'positive')):
        os.mkdir(os.path.join('test', 'positive'))
    if not os.path.exists(os.path.join('test', 'negative')):
        os.mkdir(os.path.join('test', 'negative'))
    if not os.path.exists(os.path.join('test', 'mask')):
        os.mkdir(os.path.join('test', 'mask'))
    
    test_cancer_filenames = [
        '1.png',
    ]
    test_non_cancer_filenames = [
        '2.png',
    ]

    negative_mask = np.zeros((2048, 2048), dtype=np.uint8)
    for filename in test_cancer_filenames:
        shutil.move(os.path.join('positive_png', filename), os.path.join('test', 'positive', filename))
        shutil.move(os.path.join('mask_png', filename), os.path.join('test', 'mask', filename))
    for filename in test_non_cancer_filenames:
        shutil.move(os.path.join('negative_png', filename), os.path.join('test', 'negative', filename))
        cv2.imwrite(os.path.join('test', 'mask', filename), negative_mask)

if __name__ == '__main__':
    test_image_split('test')