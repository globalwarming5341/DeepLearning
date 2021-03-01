import numpy as np
import cv2
import os
import matplotlib as plt


ROOT_PATH = './gastric'
CANCER_IMAGE_PATH = 'positive'
NON_CANCER_IMAGE_PATH = 'negative'
MASK_PATH = 'mask_png'
SAVE_PATH = 'save_path'

if __name__ == '__main__':
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        
    cancer_files = os.listdir(CANCER_IMAGE_PATH)
    non_cancer_files = os.listdir(NON_CANCER_IMAGE_PATH)
    for i, cancer_file in enumerate(cancer_files):
        cancer_image = cv2.imread(os.path.join(CANCER_IMAGE_PATH, cancer_file), cv2.IMREAD_COLOR)
        cancer_image_filename = os.path.splitext(cancer_file)[0]
        mask = cv2.imread(os.path.join(MASK_PATH, cancer_file), cv2.IMREAD_GRAYSCALE)
        for non_cancer_file in non_cancer_files:
            non_cancer_image = cv2.imread(os.path.join(NON_CANCER_IMAGE_PATH, non_cancer_file), cv2.IMREAD_COLOR)
            non_cancer_image[mask == 255] = cancer_image[mask == 255]
            cv2.imwrite(os.path.join(SAVE_PATH, '{}_{}'.format(cancer_image_filename, non_cancer_file)), non_cancer_image)
            print(non_cancer_file)
        print(i)