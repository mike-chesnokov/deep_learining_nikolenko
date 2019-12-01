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


class BNModel1(Model):
    """
    Model for Chapter 4.3 (with batch normalization)
    """
    def __init__(self, input_size=784, fc1_size=100, bn_size=100, fc2_size=100, fc3_size=10):
        super(BNModel1, self).__init__()
        
        self.W_1 = tf.Variable(tf.random.truncated_normal([input_size, fc1_size], stddev=0.1))
        self.b_1 = tf.Variable(tf.random.truncated_normal([fc1_size], stddev=0.1))
        
        self.beta = tf.Variable(tf.zeros([bn_size]))
        self.scale = tf.Variable(tf.ones([bn_size]))
        
        self.W_2 = tf.Variable(tf.random.truncated_normal([bn_size, fc2_size], stddev=0.1))
        self.b_2 = tf.Variable(tf.random.truncated_normal([fc2_size], stddev=0.1))       
        
        self.W_3 = tf.Variable(tf.random.truncated_normal([fc2_size, fc3_size], stddev=0.1))
        self.b_3 = tf.Variable(tf.random.truncated_normal([fc3_size], stddev=0.1))  
        
    def __call__(self, X):
        # input layer
        h1 = tf.nn.tanh(tf.matmul(X, self.W_1) + self.b_1)
        
        # batch norm layer
        batch_mean, batch_var = tf.nn.moments(h1, [0])
        h1_bn = tf.nn.batch_normalization(h1, batch_mean, batch_var, self.beta, self.scale, 0.001)
        
        # hidden layer
        h2 = tf.nn.tanh(tf.matmul(h1_bn, self.W_2) + self.b_2)
        
        # output layer
        return tf.nn.tanh(tf.matmul(h2, self.W_3) + self.b_3)
        #return tf.matmul(h2, self.W_3) + self.b_3


class BNModel2(Model):
    """
    Model for Chapter 4.3 (without batch normalization)
    """
    def __init__(self, input_size=784, fc1_size=100, fc2_size=100, fc3_size=10):
        super(BNModel2, self).__init__()
        
        self.W_1 = tf.Variable(tf.random.truncated_normal([input_size, fc1_size], stddev=0.1))
        self.b_1 = tf.Variable(tf.random.truncated_normal([fc1_size], stddev=0.1))
        
        self.W_2 = tf.Variable(tf.random.truncated_normal([fc1_size, fc2_size], stddev=0.1))
        self.b_2 = tf.Variable(tf.random.truncated_normal([fc2_size], stddev=0.1))       
        
        self.W_3 = tf.Variable(tf.random.truncated_normal([fc2_size, fc3_size], stddev=0.1))
        self.b_3 = tf.Variable(tf.random.truncated_normal([fc3_size], stddev=0.1))  
        
    def __call__(self, X):
        # input layer
        h1 = tf.nn.tanh(tf.matmul(X, self.W_1) + self.b_1)
                
        # hidden layer
        h2 = tf.nn.tanh(tf.matmul(h1, self.W_2) + self.b_2)
        
        # output layer
        return tf.nn.tanh(tf.matmul(h2, self.W_3) + self.b_3)
        #return tf.matmul(h2, self.W_3) + self.b_3
    

class SimpleModel(Model):
    """
    Model for Chapter 4.5 (simple model)
    """
    __name__ = 'SimpleModel'
    
    def __init__(self, input_size=784, fc1_size=300, fc2_size=100, fc3_size=10):
        super(SimpleModel, self).__init__()
        
        self.W_1 = tf.Variable(tf.random.truncated_normal([input_size, fc1_size], stddev=0.1))
        self.b_1 = tf.Variable(tf.random.truncated_normal([fc1_size], stddev=0.1))
        
        self.W_2 = tf.Variable(tf.random.truncated_normal([fc1_size, fc2_size], stddev=0.1))
        self.b_2 = tf.Variable(tf.random.truncated_normal([fc2_size], stddev=0.1))       
        
        self.W_3 = tf.Variable(tf.random.truncated_normal([fc2_size, fc3_size], stddev=0.1))
        self.b_3 = tf.Variable(tf.random.truncated_normal([fc3_size], stddev=0.1))  
        
    def __call__(self, X):
        # input layer
        h1 = tf.nn.relu(tf.matmul(X, self.W_1) + self.b_1)

        # hidden layer
        h2 = tf.nn.relu(tf.matmul(h1, self.W_2) + self.b_2)
        
        # output layer
        return tf.nn.relu(tf.matmul(h2, self.W_3) + self.b_3)
    

class BNModel(Model):
    """
    Model for Chapter 4.5 (model with Batch Normalization layer)
    """
    __name__ = 'BNModel'
    
    def __init__(self, input_size=784, fc1_size=300, fc2_size=100, fc3_size=10):
        super(BNModel, self).__init__()
        
        self.W_1 = tf.Variable(tf.random.truncated_normal([input_size, fc1_size], stddev=0.1))
        self.b_1 = tf.Variable(tf.random.truncated_normal([fc1_size], stddev=0.1))
        
        self.beta = tf.Variable(tf.zeros([fc1_size]))
        self.scale = tf.Variable(tf.ones([fc1_size]))
        
        self.W_2 = tf.Variable(tf.random.truncated_normal([fc1_size, fc2_size], stddev=0.1))
        self.b_2 = tf.Variable(tf.random.truncated_normal([fc2_size], stddev=0.1))       
        
        self.W_3 = tf.Variable(tf.random.truncated_normal([fc2_size, fc3_size], stddev=0.1))
        self.b_3 = tf.Variable(tf.random.truncated_normal([fc3_size], stddev=0.1))  
        
    def __call__(self, X):
        # input layer
        h1 = tf.nn.relu(tf.matmul(X, self.W_1) + self.b_1)
        
        # batch norm layer
        batch_mean, batch_var = tf.nn.moments(h1, [0])
        h1_bn = tf.nn.batch_normalization(h1, batch_mean, batch_var, self.beta, self.scale, 0.001)
        
        # hidden layer
        h2 = tf.nn.relu(tf.matmul(h1_bn, self.W_2) + self.b_2)
        
        # output layer
        return tf.nn.relu(tf.matmul(h2, self.W_3) + self.b_3)
    

class XavierModel(Model):
    """
    Model for Chapter 4.5 (model with Xavier init)
    """
    __name__ = 'XavierModel'
        
    def __init__(self, input_size=784, fc1_size=300, fc2_size=100, fc3_size=10):
        super(XavierModel, self).__init__()
        
        initializer = tf.initializers.GlorotUniform()
        
        self.W_1 = tf.Variable(initializer(shape=(input_size, fc1_size)))
        self.b_1 = tf.Variable(initializer(shape=(fc1_size,)))
        
        self.W_2 = tf.Variable(initializer(shape=(fc1_size, fc2_size)))
        self.b_2 = tf.Variable(initializer(shape=(fc2_size,)))       
        
        self.W_3 = tf.Variable(initializer(shape=(fc2_size, fc3_size)))
        self.b_3 = tf.Variable(initializer(shape=(fc3_size,)))  
        
    def __call__(self, X):
        # input layer
        h1 = tf.nn.relu(tf.matmul(X, self.W_1) + self.b_1)
        
        # hidden layer
        h2 = tf.nn.relu(tf.matmul(h1, self.W_2) + self.b_2)
        
        # output layer
        return tf.nn.relu(tf.matmul(h2, self.W_3) + self.b_3)
    

class XavierBNModel(Model):
    """
    Model for Chapter 4.5 (model with Batch Normalization layer and Xavier initialization)
    """
    __name__ = 'XavierBNModel'
    
    def __init__(self, input_size=784, fc1_size=300, fc2_size=100, fc3_size=10):
        super(XavierBNModel, self).__init__()
        
        initializer = tf.initializers.GlorotUniform()
        
        self.W_1 = tf.Variable(initializer(shape=(input_size, fc1_size)))
        self.b_1 = tf.Variable(initializer(shape=(fc1_size,)))
        
        self.beta = tf.Variable(tf.zeros([fc1_size]))
        self.scale = tf.Variable(tf.ones([fc1_size]))
        
        self.W_2 = tf.Variable(initializer(shape=(fc1_size, fc2_size)))
        self.b_2 = tf.Variable(initializer(shape=(fc2_size,)))       
        
        self.W_3 = tf.Variable(initializer(shape=(fc2_size, fc3_size)))
        self.b_3 = tf.Variable(initializer(shape=(fc3_size,)))  
        
    def __call__(self, X):
        # input layer
        h1 = tf.nn.relu(tf.matmul(X, self.W_1) + self.b_1)
        
        # batch norm layer
        batch_mean, batch_var = tf.nn.moments(h1, [0])
        h1_bn = tf.nn.batch_normalization(h1, batch_mean, batch_var, self.beta, self.scale, 0.001)
        
        # hidden layer
        h2 = tf.nn.relu(tf.matmul(h1_bn, self.W_2) + self.b_2)
        
        # output layer
        return tf.nn.relu(tf.matmul(h2, self.W_3) + self.b_3)
