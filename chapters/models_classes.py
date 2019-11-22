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
    
    
class HidLayerModel(Model):
    """
    Classification Model for Chapter 3.6 (nn with hidden layer and dropout)
    """
    def __init__(self, W_relu, b_relu, W_logit, b_logit, p_keep=0.5):
        super(HidLayerModel, self).__init__()
        self.W_relu = W_relu
        self.b_relu = b_relu
        self.W_logit = W_logit
        self.b_logit = b_logit
        self.p_keep = p_keep
        
    def call(self, X):
        hidden_layer = tf.nn.relu(tf.matmul(X, self.W_relu) + self.b_relu)  
        h_drop = tf.nn.dropout(hidden_layer, self.p_keep)
        
        return tf.matmul(h_drop, self.W_logit) + self.b_logit
    