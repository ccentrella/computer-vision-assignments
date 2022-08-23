import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''The basic summation function'''
def sum_values(w,X,b):
    return np.dot(w, X) + b

'''The step activation function'''
def step_function(x):
    if x < 0:
        return 0
    else:
        return 1

'''The sigmoid logistic activation function'''
def signma_function(x):
    return 1/(1 + np.exp(-x))

'''The softmax logistic activation function'''
def softmax_function(list):
    return (np.exp(list)/np.exp(list).sum())

def softmax_function2(list):
    return np.softmax(list)

'''The tanh activation function'''
def tanh_function(x):
    return np.tanh(x)

'''The ReLU activation function'''
def relu_function(x):
    if x < 0:
        return 0
    else:
        return x

'''Leaky ReLU activation function'''
def l_relu_function(x):
    if x < 0:
        return -0.01 * x
    else:
        return x

# Get linear combination
z = sum_values([2,2,2,2,2,2,2],[2142,235,234,234,234,234,234],100)
print (z)

# Determine whether the data passes
print (step_function(z))

# Build simple 5x5x3 layer model
model = keras.Sequential([
    layers.Dense(5, input_dim=4), layers.Dense(5), layers.Dense(3),
])
print (model.summary())

# Build simple 20x20x10x10x5x2 layer model
model2 = keras.Sequential([
    layers.Dense(20, input_dim=5), layers.Dense(20), layers.Dense(10),
    layers.Dense(10), layers.Dense(5), layers.Dense(2)
])
print (model2.summary())

# Build simple model with 5 hidden layers and 10 neurons each
model3 = keras.Sequential([
    layers.Dense(20,input_dim=3), layers.Dense(10), layers.Dense(10),
     layers.Dense(10), layers.Dense(10), layers.Dense(5)
])
print (model3.summary())
