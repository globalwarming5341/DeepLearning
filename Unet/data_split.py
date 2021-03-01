import shutil
import os
import random
import glob

DATA_PATH = './data_segmentation'


def move_files(data_files_from, to):
    size = len(data_files_from)
    if not os.path.exists(to):
        os.mkdir(to)
    for i, f in enumerate(data_files_from):
        shutil.move(f, to)
        print('{}/{}'.format(i + 1, size))
    

def split(data_path, train_ratio=0.9):
    
    pos_data_files = glob.glob(os.path.join(data_path, 'pos_*.npy'))
    neg_data_files = glob.glob(os.path.join(data_path, 'neg_*.npy'))
    random.shuffle(pos_data_files)
    random.shuffle(neg_data_files)

    pos_boundary = int(len(pos_data_files) * train_ratio)
    neg_boundary = int(len(neg_data_files) * train_ratio)

    train_pos_data_files = pos_data_files[:pos_boundary]
    train_neg_data_files = neg_data_files[:neg_boundary]

    val_pos_data_files = pos_data_files[pos_boundary:]
    val_neg_data_files = neg_data_files[neg_boundary:]

    move_files(train_pos_data_files, os.path.join(data_path, 'train'))
    move_files(train_neg_data_files, os.path.join(data_path, 'train'))
    move_files(val_pos_data_files, os.path.join(data_path, 'val'))
    move_files(val_neg_data_files, os.path.join(data_path, 'val'))

if __name__ == '__main__':
    split(DATA_PATH)




        