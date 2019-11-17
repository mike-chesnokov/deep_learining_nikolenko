# Losses using TF2
import numpy as np
import tensorflow as tf


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss using TF2
    """
    return tf.reduce_sum(tf.square(y_true - y_pred))

def cross_entropy_loss(y_true, y_pred):
    """
    Cross entropy loss using TF2.
    
    y_true and y_pred - ohe vectors
    """
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log1p(y_pred), 1))
