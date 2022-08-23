from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, ReLU, Softmax, BatchNormalization, concatenate, AveragePooling2D, Dropout, Flatten, Input
from tensorflow.keras.initializers import GlorotUniform, Constant
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt

# Create file path for model
file_path = f'D:\\Keras\\Logs\\Inception\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
makedirs(file_path, exist_ok=True)

# Load data
print('Fetching data...')
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Verify results
print('Verifying data...')
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000,1)
assert y_test.shape == (10000, 1)

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
    filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
    
    # 1x1 route
    conv_1x1 = Conv2D(filters_1x1, kernel_size=(1,1), padding='same', activation='relu',
    kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    # 3x3 route
    pre_conv_3x3 = Conv2D(filters_3x3_reduce, kernel_size=(1,1), padding='same', activation='relu',
    kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
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

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output

# Create convolutional layers
#   Conv1
inputs = Input(shape=(32,32,3))
conv1 = Conv2D(64, kernel_size=(7,7), padding='same', strides=(2,2), activation='relu',
name='conv_1_7_x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init, input_shape=(32,32,3))(inputs)
conv1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name='max_pool_1_3x3/2')(conv1)
conv1 = BatchNormalization()(conv1)

#   Conv2
conv2 = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(conv1)

#   Conv3
conv3 = Conv2D(192, kernel_size=(3,3), padding='same', activation='relu')(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(conv3)

# Create inception modules
#   IM4,5
im4_5 = inception_module(conv3, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)

#   IM6,7
im6_7 = inception_module(im4_5, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)
im6_7 = MaxPool2D(pool_size=(3,3),strides=(2,2), padding='same')(im6_7)

#   IM8,9
im8_9 = inception_module(im6_7, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)

#   IM10,11
im10_11 = inception_module(im8_9, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)

#   IM12,13
im12_13 = inception_module(im10_11, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)

#   IM14,15
im14_15 = inception_module(im12_13, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64)

#   IM16,17
im16_17 = inception_module(im14_15, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
im16_17 = MaxPool2D(pool_size=(3,3),strides=(2,2), padding='same')(im16_17)

#   IM18,19
im18_19 = inception_module(im16_17, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)

#   IMG20,21
im20_21 = inception_module(im18_19, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384,
filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128)

# Create classification layers
#   Pooling and flatten layer
#flatten = AveragePooling2D(pool_size=(7,7), strides=(1,1), padding='valid')(im20_21)
flatten = Flatten()(im20_21)
flatten = Dropout(0.4)(flatten)

#   FC22
outputs = Dense(100)(flatten)
outputs = Softmax()(outputs)

#   Create model instance
model = Model(inputs=inputs, outputs=outputs)

# Print summary
model.summary()

#   Configure data augmentation generator
print('Configuring data augmentation generator...')
datagen = ImageDataGenerator(
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    shear_range=0.1,
    rotation_range=20,
    zoom_range=0.1
)
datagen.fit(x_train)

# Compile model
print('Compiling model...')

#   Required variables
batch_size = 256
epochs = 90
initial_lr = 0.01
loss = CategoricalCrossentropy()
lr_decay = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=(8*50000/batch_size), decay_rate=0.96)
optimizer = SGD(learning_rate=lr_decay, momentum=0.9, nesterov = False)

checkpoint = ModelCheckpoint(f'{file_path}Weights.h5', monitor='val_loss', save_best_only=True, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

#   Compile
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

#   Begin training
print('Training model...')
hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, verbose=1, callbacks=(
early_stopping, checkpoint), validation_data=(x_test, y_test))
print('Training complete.')

#   View Test Results
print('Running model on test data...')
score = 100*model.evaluate(x_test, y_test, verbose=1)
print('Test complete.')
print('Test Accuracy: %.4f%%' % score[1])

#   Accuracy
print('Loading statistics...')
ax = plt.subplot()
ax.plot(hist.history['accuracy'], 'o-', label='Training Data')
ax.plot(hist.history['val_accuracy'], 'x-', label='Testing Data')
ax.legend(loc=0)
ax.set_title('Training/val accuracy per epoch')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
plt.savefig(f'{file_path}Results-Accuracy.png')
plt.show()

#   Loss
ax = plt.subplot()
ax.plot(hist.history['loss'], 'o-', label='Training Data')
ax.plot(hist.history['val_loss'], 'x-', label='Testing Data')
ax.legend(loc=0)
ax.set_title('Training/val loss per epoch')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.savefig(f'{file_path}Results-Loss.png')
plt.show()

print('Complete.')