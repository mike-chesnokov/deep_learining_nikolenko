# Classes for models
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


class RegModel():
    """
    Regression Model for Chapter 2.6
    """
    def __init__(self, k, b):
        self.k = k
        self.b = b
        
    def __call__(self, X):
        return tf.matmul(X, self.k) + self.b

    
class LogRegModel(Model):
    """
    Classification Model for Chapter 3.6 (simple logreg)
    """
    def __init__(self, W, b):
        super(LogRegModel, self).__init__()
        self.W = W
        self.b = b
        
    def call(self, X):
        return tf.matmul(X, self.W) + self.b
    