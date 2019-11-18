import os
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def _fix_seeds(seed=42):
    """
    Fix seeds for reproducible results
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
def plot_metric_history(metric_history, title='Loss', xlabel='Epoch', ylabel='Loss'):
    """
    Visualization of metrics history during train and test
    
    metric_history: dict of 2 lists, keys: 'train', 'test'
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    ax.plot(metric_history['train'], label='train', alpha=0.5)
    ax.plot(metric_history['test'], label='test', alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()
    

def show_images(data,
                data_labels,
                random_sample=True,
                seed=42,
                sample_size=20,
                num_cols=4,
                image_ids=None):
    """
    Show images from MNIST dataset.
    
    Parameters:
        :param data: numpy array with mnist digits
        :param data_labels: numpy array with mnist labels
        :param random_sample: bool, show random pictures or not
        :param sample_size: int, how many pictures to select randomly
        :param image_ids: list, image ids if `random_sample` = False
        :param seed: int, seed for random image taking
    """
    if random_sample:
        np.random.seed(seed)
        image_ids = np.random.choice(list(range(len(data))), sample_size)
    else:
        if image_ids is None:
            raise ValueError('Provide "image_ids" list')
    
    image_ids_num = len(image_ids)
    num_rows = int(np.ceil(image_ids_num/num_cols))
    
    fig, ax = plt.subplots(figsize=(2 * num_cols, 2 * num_rows), ncols=num_cols, nrows=num_rows)
    for ind, img_id in enumerate(image_ids):
        img = data[img_id]
        
        col_ind = ind%num_cols
        row_ind = int(np.floor(ind/num_cols))
        
        if image_ids_num > num_cols: # if row num more than 1
            ax[row_ind, col_ind].set_title('Class=' + str(data_labels[img_id]))
            ax[row_ind, col_ind].imshow(img, cmap=plt.cm.binary)
        else: # if only 1 row
            ax[ind].set_title('Class=' + str(data_labels[img_id]))
            ax[ind].imshow(img)
            
    plt.tight_layout()  
    plt.show()    
    