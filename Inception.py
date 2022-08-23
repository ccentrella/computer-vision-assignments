from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, ReLU, Softmax, BatchNormalization, Concatenate, AveragePooling2D
from tensorflow.keras.initializers import GlorotUniform, Constant
import time
from os import makedirs
import numpy as np
from tensorflow.python.keras.layers.core import Dropout, Flatten

# Create file path for model
file_path = f'D:\\Keras\\Logs\\LeNet-5\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
makedirs(file_path, exist_ok=True)

# Load data
print('Loading data...')
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Prepare data
print('Preparing data...')

#   Normalize images
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean)/(std + 1e-7)
x_test = (x_test - mean)/(std + 1e-7)

#   One-hot encode labels
num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build model
print('Creating model...')
kernel_init = GlorotUniform()
bias_init = Constant(0.2)

# Defines an inception module
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3,
    filters_5x5_reduce, filters_5x5, filters_pool_proj, name='None'):
    
    # 1x1 route
    conv_1x1 = Conv2D(filters_1x1, kernel_size=(1,1), padding='same', activation='relu',
    kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    # 3x3 route
    pre_conv_3x3 = Conv2D(filters_3x3_reduce, kernel_size=(1,1), padding='same', activation='relu',
    kernel_init=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, kernel_size=(3,3), padding='same', activation='relu',
    kernel_initializer=kernel_init, bias_initializer=bias_init)(pre_conv_3x3)

    # 5x5 route
    pre_conv_5x5 = Conv2D(filters_5x5_reduce, kernel_size=(1,1), padding='same', activation='relu',
    kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, kernel_size=(5,5), padding='same', activation='relu',
    kernel_initializer=kernel_init, bias_initializer=bias_init)(pre_conv_5x5)

    # Max-pooling
    pool_proj = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, kernel_size=(1,1), padding='same', activation='relu',
    kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = Concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output

model = Sequential()

# Create convolutional layers
#   Conv1
model.add(Conv2D(64, kernel_size=(7,7), padding='same', strides=(2,2), activation='relu',
name='conv_1_7_x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init, input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name='max_pool_1_3x3/2'))
model.add(BatchNormalization())

#   Conv2
model.add(Conv2D(64, kernel_size=(1,1), padding='same', activation='relu'))

#   Conv3
model.add(Conv2D(192, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same'))

# Create inception modules
#   IM4,5
x = model.get_input_at(1)
model.add(inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32))

#   IM6,7
model.add(inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2), padding='same'))

#   IM8,9
model.add(inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64))

#   IM10,11
model.add(inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64))

#   IM12,13
model.add(inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64))

#   IM14,15
model.add(inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64))

#   IM16,17
model.add(inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2), padding='same'))

#   IM18,19
model.add(inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128))

#   IMG20,21
model.add(inception_module(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384,
filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128))

# Create classification layers
#   Average and flatten layer
model.add(AveragePooling2D(pool_size=(7,7), strides=(1,1), padding='valid'))
model.add(Flatten())
model.add(Dropout(0.4))

#   FC22
model.add(Dense(100))
model.add(Softmax())