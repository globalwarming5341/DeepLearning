import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse


ROOT_PATH = './gastric/'
CANCER_IMAGE_PATH = 'positive_png'
NON_CANCER_IMAGE_PATH = 'negative_png'
MASK_PATH = 'mask_png'

class DataGenerator(object):
    def __init__(self,
                 cancer_path,
                 non_cancer_path,
                 mask_path,
                 *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)
        self._cancer_path = cancer_path
        self._non_cancer_path = non_cancer_path
        self._mask_path = mask_path

    def _rotate_image(self, image, angle):
        pass

    def _image_augment(self, image, option='all'):
        if isinstance(option, int):
            return [
                np.fliplr,
                np.flipud,
                lambda x: np.rot90(x, k=1),
                lambda x: np.rot90(x, k=2),
                lambda x: np.rot90(x, k=3),
            ][option](image)
        else:
            if option == 'all':
                return [
                    np.rot90(image, k=1),
                    np.rot90(image, k=2),
                    np.rot90(image, k=-1),
                    np.fliplr(image),
                    np.flipud(image)
                ]
            elif option == 'rotation':
                return [
                    np.rot90(image, k=1),
                    np.rot90(image, k=2),
                    np.rot90(image, k=3),
                ]
            elif option == 'flip':
                return [
                    np.fliplr(image),
                    np.flipud(image)
                ]
            else:
                return []
    
    def crop_for_classification(self, num, crop_width, crop_height, crop_strides, save_path, beta):
        cancer_images = os.listdir(self._cancer_path)
        non_cancer_images = os.listdir(self._non_cancer_path)
        cancer_image_len = len(cancer_images)
        non_cancer_image_len = len(non_cancer_images)
        save_index = 1
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i, filename in enumerate(cancer_images):
            image = cv2.imread(os.path.join(self._cancer_path, filename), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            label = cv2.imread(os.path.join(self._mask_path, filename), cv2.IMREAD_GRAYSCALE)
            crop_size = crop_width * crop_height
            x = 0
            y = 0
            while True:
                crop_label = label[y: y + crop_height, x: x + crop_width]
                crop_label[crop_label > 0] = 1
                area_cancer = crop_label.sum()
                if area_cancer / crop_size >= beta:
                    crop_image = image[y: y + crop_height, x: x + crop_width, :]
                    np.save(os.path.join(save_path, 'pos_{}_{}.npy'.format(filename, save_index)), crop_image)
                    save_index += 1
                    # for aug_data in self._image_augment(save_data):
                    #     np.save(os.path.join(save_path, 'pos_aug_{}.npy'.format(save_index)), aug_data)
                    #     save_index += 1       
                x += crop_strides
                if x + crop_width > image_width:
                    x = 0
                    y += crop_strides
                    if y + crop_height > image_height:
                        break
            print('{}/{} {}'.format(i + 1, cancer_image_len, save_index))    

        data_files = os.listdir(save_path)
        random.shuffle(data_files)
        data_size = len(data_files)
        idx = 0
        while save_index <= num:
            k = 0
            data = np.load(os.path.join(save_path, data_files[idx]))
            np.save(os.path.join(save_path, '{}'.format(data_files[idx].replace('pos_', 'pos_aug_'), save_index)), self._image_augment(data, option=k))
            save_index += 1
            idx += 1
            if idx >= data_size:
                idx = 0
                k += 1
                if k >= 5:
                    
                    break

        print(save_index)
        save_index = 1
        for i, filename in enumerate(non_cancer_images):
            image = cv2.imread(os.path.join(self._non_cancer_path, filename), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            x = 0
            y = 0
            while True:
                crop_image = image[y: y + crop_height, x: x + crop_width, :]
                np.save(os.path.join(save_path, 'neg_{}_{}.npy'.format(filename, save_index)), crop_image)
                save_index += 1

                x += crop_strides
                if x + crop_width > image_width:
                    x = 0
                    y += crop_strides
                    if y + crop_height > image_height:
                        break
            print('{}/{} {}'.format(i + 1, non_cancer_image_len, save_index))
        
        data_files = os.listdir(save_path)
        data_files = list(filter(lambda x: x.startswith('neg'), data_files))
        random.shuffle(data_files)
        data_size = len(data_files)
        idx = 0
        thres = crop_width * crop_height * 0.94
        while save_index <= num:
            k = 0
            data = np.load(os.path.join(save_path, data_files[idx]))
            image_gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            image_gray[image_gray >= 253] = 1
            image_gray[image_gray < 253] = 0
            if image_gray.sum() <= thres:
                np.save(os.path.join(save_path, '{}'.format(data_files[idx].replace('neg_', 'neg_aug_'), save_index)), self._image_augment(data, option=k))
                save_index += 1
            idx += 1
            if idx >= data_size:
                idx = 0
                k += 1
                if k >= 5:
                    break

        print(save_index)
    
    def crop_for_segmentation(self, crop_width, crop_height, crop_strides, save_path, beta=0.5, augment=True, dilate=False):
        cancer_images = os.listdir(self._cancer_path)
        non_cancer_images = os.listdir(self._non_cancer_path)
        cancer_image_len = len(cancer_images)
        non_cancer_image_len = len(non_cancer_images)
        save_index = 1
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i, filename in enumerate(cancer_images):
            basename = os.path.splitext(filename)[0]
            image = cv2.imread(os.path.join(self._cancer_path, filename), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            original_label = cv2.imread(os.path.join(self._mask_path, filename), cv2.IMREAD_GRAYSCALE)
            label = np.zeros(shape=original_label.shape, dtype=np.uint8)
            if dilate:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (128, 128))
                dilated_label = cv2.dilate(original_label, kernel)
                label[dilated_label == 255] = 1
                

            label[original_label == 255] = 2
            
            crop_size = crop_width * crop_height
            x = 0
            y = 0
            while True:
                crop_label = label[y: y + crop_height, x: x + crop_width]
                area_total = np.where(crop_label > 0, 1, 0).sum()
                area_cancer = np.where(crop_label == 2, 1, 0).sum()
                if area_total / crop_size > beta and area_cancer / area_total >= beta - 0.05:
                    crop_image = image[y: y + crop_height, x: x + crop_width, :]
                    save_data = np.concatenate([crop_image, crop_label[:, :, np.newaxis]], axis=-1)
                    np.save(os.path.join(save_path, 'pos_{}_{}_{}_{}.npy'.format(basename, x, y, save_index)), save_data)
                    save_index += 1

                    if augment:
                        for aug_data in self._image_augment(save_data):
                            np.save(os.path.join(save_path, 'pos_aug_{}_{}_{}_{}.npy'.format(basename, x, y, save_index)), aug_data)
                            save_index += 1
                x += crop_strides
                if x + crop_width > image_width:
                    x = 0
                    y += crop_strides
                    if y + crop_height > image_height:
                        break
            print('{}/{} {}'.format(i + 1, cancer_image_len, save_index))
                
        
        print(save_index)
        save_index = 1
        
        for i, filename in enumerate(non_cancer_images):
            basename = os.path.splitext(filename)[0]
            image = cv2.imread(os.path.join(self._non_cancer_path, filename), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            crop_label = np.ones(shape=(crop_height, crop_width, 1), dtype=np.uint8)
            x = 0
            y = 0
            while True:
                crop_image = image[y: y + crop_height, x: x + crop_width, :]
                save_data = np.concatenate([crop_image, crop_label], axis=-1)
                np.save(os.path.join(save_path, 'neg_{}_{}_{}_{}.npy'.format(basename, x, y, save_index)), save_data)
                save_index += 1
                image_gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
                image_gray[image_gray >= 253] = 1
                image_gray[image_gray < 253] = 0
                if augment and image_gray.sum() <= crop_width * crop_height * 0.94:
                    for aug_data in self._image_augment(save_data):
                        np.save(os.path.join(save_path, 'neg_aug_{}_{}_{}_{}.npy'.format(basename, x, y, save_index)), aug_data)
                        save_index += 1

                x += crop_strides
                if x + crop_width > image_width:
                    x = 0
                    y += crop_strides
                    if y + crop_height > image_height:
                        break
            print('{}/{} {}'.format(i + 1, non_cancer_image_len, save_index))
        print(save_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer_path', help="path to cancer images.", required=True)
    parser.add_argument('--non_cancer_path', help="path to non-cancer images.", required=True)
    parser.add_argument('--mask_path', help="path to binary mask images.", required=True)
    parser.add_argument('--save_path', help="path to binary mask images.", required=True)
    parser.add_argument('--augment', action='store_true', help="Enable augumentation to generate data.")
    parser.add_argument('--dilate', action='store_true', help="Enable dilation to generate data.")
    parser.add_argument('--beta', help="Threshold for cropping cancer area.", type=float, default=0.2)
    args = parser.parse_args()
    generator = DataGenerator(args.cancer_path, args.non_cancer_path, args.mask_path)
    generator.crop_for_segmentation(512, 512, 256, os.path.join(args.save_path), beta=args.beta, augment=args.augment, dilate=args.dilate)