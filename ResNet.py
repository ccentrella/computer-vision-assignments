# Import required dependencies
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, ReLU, Softmax, Add, Input, Flatten, MaxPool2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
import time
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt

# Create file path for model
file_path = f'D:\\Keras\\Logs\\ResNet\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
makedirs(file_path, exist_ok=True)

# Load data
print('Fetching data...')
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Verify data
print('Verifying data...')
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# Prepare data
print('Preparing data...')

#   Normalize images
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean)/(std+1e-7)
x_test = (x_test - mean)/(std+1e-7)

#   One-hot encode labels
num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build model
print('Creating model...')

# Create a ResNet residual module
def bottleneck_residual_block(x, kernel_size, filters, reduce=False,s=2):
    F1, F2, F3 = filters

    # The shortcut path, which is equal to the input
    x_shortcut = x

    # Configure the module based on whether we use a regular shortcut or reduce shortcut
    if reduce:
        x_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), kernel_regularizer=L2(0.0003))(x_shortcut)
        x_shortcut = BatchNormalization(axis=3)(x_shortcut)
        
        # First component of main path
        x = Conv2D(filters = F1, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_regularizer=L2(0.0003))(x)
        x = BatchNormalization(axis=3)(x)
        x = ReLU()(x)

    else:
        # First component of main path
        x = Conv2D(filters = F1, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_regularizer=L2(0.0003))(x)
        x = BatchNormalization(axis=3)(x)
        x = ReLU()(x)

    # Second component of main path
    x = Conv2D(filters = F2, kernel_size=kernel_size, strides=(1,1), padding='same', kernel_regularizer=L2(0.0003))(x)
    x = BatchNormalization(axis = 3)(x)
    x = ReLU()(x)

    # Third component of main path
    x = Conv2D(filters = F3, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_regularizer=L2(0.0003))(x)
    x = BatchNormalization(axis=3)(x)

    # Final step
    x = Add() ([x, x_shortcut])
    x = ReLU()(x)

    return x 

# Creates a new ResNet model, with 50 layers and outputting to the specified number of classes
def ResNet50(input_shape, classes):
    x_input = Input(input_shape)

    #   Stage 1
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), name='conv1', kernel_regularizer=L2(0.0003))(x_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

    #   Stage 2
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[64, 64, 256], reduce=True, s=1)
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[64, 64, 256])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[64, 64, 256])

    #   Stage 3
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[128, 128, 512], reduce=True, s=2)
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[128, 128, 512])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[128, 128, 512])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[128, 128, 512])

    #   Stage 4
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[256, 256, 1024], reduce=True, s=2)
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[256, 256, 1024])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[256, 256, 1024])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[256, 256, 1024])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[256, 256, 1024])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[256, 256, 1024])

    #   Stage 5
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[512, 512, 2048], reduce=True, s=2)
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[512, 512, 2048])
    x = bottleneck_residual_block(x, kernel_size=(3,3), filters=[512, 512, 2048])

    #   Classification
    x = AveragePooling2D(pool_size=(1,1))(x)
    x = Flatten()(x)
    x = Dense(classes, name='fc1')(x)
    x = Softmax()(x)

    model = Model(inputs = x_input, outputs = x, name='ResNet50')
    return model

#   Create model
model = ResNet50(input_shape=(32, 32, 3), classes=100)

# Print summary
model.summary()

# Configure data augmentation generator
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
epochs = 200
batch_size = 256
initial_lr = 0.01
loss = CategoricalCrossentropy()
optimizer = SGD(learning_rate=initial_lr, momentum=0.9, nesterov=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.5e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint(f'{file_path}Weights.h5', monitor='val_loss', verbose=1, save_best_only=True)

model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

# Train the model
print('Beginning training...')
hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, 
    validation_data = (x_test, y_test), callbacks=[reduce_lr, early_stopping, checkpoint], verbose=1)
print('Training complete.')

# View test results
print('Running model on test data...')
score = 100*model.evaluate(x_test, y_test, verbose=1)
print('Test complete.')
print(f'Test Accuracy: {score[1]}')

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