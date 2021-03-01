from unet import Unet
from unet_metrics import get_metrics
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    unet = Unet(weight_path='model.hdf5', residual=False)

    test_path = './data_segmentation_test'
    if not os.path.exists('prediction'):
        os.mkdir('prediction')

    test_files = os.listdir(test_path)
    test_files = list(filter(lambda x: x.startswith('p'), test_files))
    test_size = len(test_files)

    metrics = {
        'precision': 0.,
        'recall': 0.,
        'accuracy': 0.,
        'iou': 0.
    }

    for i, test_file in enumerate(test_files):
        # prediction = unet.predict_from_image(os.path.join(test_path, test_file))
        data = np.load(os.path.join(test_path, test_file))
        
        image = data[:, :, :3].astype(np.float32)
        label = data[:, :, 3]
        label[label != 2] = 0
        label[label == 2] = 1
        label = label.astype(np.float32)
        pred = unet.predict_proba(image)


        m = get_metrics(pred, label, 0.15)

        if np.isnan(m['precision']):
            continue
        else:
            metrics['precision'] += m['precision']
            metrics['recall'] += m['recall']
            metrics['accuracy'] += m['accuracy']
            metrics['iou'] += m['iou']
            precision = m['precision']
            recall = m['recall']
            accuracy = m['accuracy']
            iou = m['iou']

        plt.figure(figsize=(16,16))
        plt.suptitle('precision: {} recall: {} accuracy: {} IoU: {}'.format(precision, recall, accuracy, iou))
        plt.subplot(2, 2, 1)
        plt.title('Original')
        plt.imshow(image.astype(np.uint8))
        plt.subplot(2, 2, 2)
        plt.title('Label')
        plt.imshow(label, cmap='gray')
        plt.subplot(2, 2, 3)
        plt.title('Prediction')
        plt.imshow(pred, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.title('Binary (threshold = 0.5)')

        binary = pred.copy()
        binary[binary >= 0.15] = 1.
        binary[binary < 0.15] = 0

        plt.imshow(binary, cmap='gray')
        plt.savefig('./unet_pred/{}.png'.format(test_file))
        plt.close('all')

        

        print('{}/{}'.format(i + 1, test_size))
        # prediction = prediction * 255.
        # prediction = prediction.astype(np.uint8)
        # cv2.imwrite(os.path.join('prediction', test_file), prediction)
        # print(test_file)
    for k in metrics:
        metrics[k] /= test_size
    print(metrics)

