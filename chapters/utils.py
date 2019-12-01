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
    plt.grid(which='both', alpha=0.3, ls='--')
    plt.show()
    

def plot_repeated_exp_metrics(loss_plain_mtrx, acc_plain_mtrx, repeat_num, num_steps, model_name):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    ax[0].plot(loss_plain_mtrx['train'].mean(axis=0), label='train mean', alpha=0.5, c='lime')
    ax[0].plot(loss_plain_mtrx['train'].max(axis=0), label='train max', alpha=0.5, c='royalblue')
    ax[0].plot(loss_plain_mtrx['train'].min(axis=0), label='train min', alpha=0.5, c='royalblue')  
    ax[0].fill_between(list(range(num_steps)), 
                     loss_plain_mtrx['train'].max(axis=0),
                     loss_plain_mtrx['train'].min(axis=0), alpha=0.1, color='royalblue')

    ax[0].set_title('{model_name} {repeat_num} times repeated avg/min/max train loss'.format(
        repeat_num=repeat_num, model_name=model_name))
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[0].plot(loss_plain_mtrx['test'].mean(axis=0), label='test mean', alpha=0.5, c='yellow')
    ax[0].plot(loss_plain_mtrx['test'].max(axis=0), label='test max', alpha=0.5, c='tomato')
    ax[0].plot(loss_plain_mtrx['test'].min(axis=0), label='test min', alpha=0.5, c='tomato')  
    ax[0].fill_between(list(range(num_steps)), 
                     loss_plain_mtrx['test'].max(axis=0),
                     loss_plain_mtrx['test'].min(axis=0), alpha=0.1, color='tomato')

    ax[0].set_title('{model_name} {repeat_num} times repeated avg/min/max test loss'.format(
        repeat_num=repeat_num, model_name=model_name))
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()    

    ax[1].axhline(y=1.0, c='grey', alpha=0.3, ls='--', lw=1)
    ax[1].axhline(y=0.9, c='grey', alpha=0.3, ls='--', lw=1)
    ax[1].axhline(y=0.8, c='grey', alpha=0.3, ls='--', lw=1)
    ax[1].axhline(y=0.7, c='grey', alpha=0.3, ls='--', lw=1)
    ax[1].axhline(y=0.6, c='grey', alpha=0.3, ls='--', lw=1)

    ax[1].plot(acc_plain_mtrx['train'].mean(axis=0), label='train mean', alpha=0.5, c='lime')
    ax[1].plot(acc_plain_mtrx['train'].max(axis=0), label='train max', alpha=0.5, c='royalblue')
    ax[1].plot(acc_plain_mtrx['train'].min(axis=0), label='train min', alpha=0.5, c='royalblue')  
    ax[1].fill_between(list(range(num_steps)), 
                     acc_plain_mtrx['train'].max(axis=0),
                     acc_plain_mtrx['train'].min(axis=0), alpha=0.1, color='royalblue')

    ax[1].set_title('{model_name} {repeat_num} times repeated avg/min/max train acc'.format(
        repeat_num=repeat_num, model_name=model_name) )
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    ax[1].plot(acc_plain_mtrx['test'].mean(axis=0), label='test mean', alpha=0.5, c='yellow')
    ax[1].plot(acc_plain_mtrx['test'].max(axis=0), label='test max', alpha=0.5, c='tomato')
    ax[1].plot(acc_plain_mtrx['test'].min(axis=0), label='test min', alpha=0.5, c='tomato')  
    ax[1].fill_between(list(range(num_steps)), 
                     acc_plain_mtrx['test'].max(axis=0),
                     acc_plain_mtrx['test'].min(axis=0), alpha=0.1, color='tomato')

    ax[1].set_title('{model_name} {repeat_num} times repeated avg/min/max test acc'.format(
        repeat_num=repeat_num, model_name=model_name) )
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.tight_layout()
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
    
    fig, ax = plt.subplots(figsize=(3 * num_cols, 3 * num_rows), ncols=num_cols, nrows=num_rows)
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
    