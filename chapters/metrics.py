# Metrics using TF2
import numpy as np
import tensorflow as tf


def accuracy(y_true, y_pred):
    """
    Multiclass accuracy
    """
    return tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_true, 1), 
                                 tf.argmax(y_pred, 1)), tf.float32)).numpy()
    