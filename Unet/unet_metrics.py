import numpy as np

def get_metrics(y_pred, y_true, threshold=0.5):
    y_pred = y_pred.copy()
    y_true = y_true.copy()
    y_pred[y_pred >= threshold] = 1.
    y_pred[y_pred < threshold] = 0.

    y_true[y_true >= threshold] = 1.
    y_true[y_true < threshold] = 0.

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    tp = np.sum(y_pred * y_true)
    fp = y_pred.sum() - tp
    fn = y_true.sum() - tp


    tn = y_pred.shape[0] - tp - fp - fn


    precision = tp / (tp + fp)



    recall = tp / (tp + fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    iou = tp / (tp + fp + fn)

    return {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'iou': iou}