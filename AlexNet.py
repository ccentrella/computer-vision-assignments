from tensorflow.keras import activations, callbacks, datasets, optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, ReLU, Softmax, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler

# File location
file_path = f'D:\\Keras\\Logs\\AlexNet\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
os.makedirs(file_path, exist_ok=True)

# Load data
#   Use 'fine' mode to have 100 output classes
print('Fetching data...')
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data(label_mode='fine')

# Verify data
print('Verifying data...')
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# Preprocess data
print('Preparing data...')

#   Image Normalization
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean)/(std + 1e-7)
x_test = (x_test - mean)/(std + 1e-7)

#   Scale images
img_cols, img_rows = 32, 32
x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 3)
x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 3)

#   One-hot encoding
num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Create model
print('Creating model...')
model = Sequential(name='AlexNet')

#   C1
model.add(Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same',
input_shape=(32, 32, 3), kernel_regularizer=L2(0.0005)))
model.add(Activation(ReLU()))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#   C2
model.add(Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=L2(0.0005)))
model.add(Activation(ReLU()))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#   C3
model.add(Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=L2(0.0005)))
model.add(Activation(ReLU()))

#   C4
model.add(Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=L2(0.0005)))
model.add(Activation(ReLU()))

#   C5
model.add(Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=L2(0.0005)))
model.add(Activation(ReLU()))

#   Flatten
model.add(Flatten())

#   FC6
model.add(Dense(units=1000, kernel_regularizer=L2(0.0005)))
model.add(Activation(ReLU()))
model.add(Dropout(0.5))

#   FC7
model.add(Dense(units=1000, kernel_regularizer=L2(0.0005)))
model.add(Activation(ReLU()))
model.add(Dropout(0.5))

#   FC8 (Output)
model.add(Dense(units=100))
model.add(Activation(Softmax()))

#   Print summary
model.summary()

# Data augmentation
print('Configuring data generator...')
datagen = ImageDataGenerator(
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    shear_range=10,
    rotation_range=0.1,
    zoom_range=0.1,
)
datagen.fit(x_train)

# Compile
print('Compiling...')

#   Required variables
batch_size = 256
epochs = 90
learning_rate = 0.001
optimizer = optimizers.Adam(learning_rate=learning_rate)
loss = losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=file_path+'Weights.h5',
     monitor='val_loss', save_best_only=True, verbose=1)

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

def get_learning_rate(epoch):
    if epoch < 20:
        lr = 0.001
    elif epoch < 50:
        lr = 0.0005
    else:
        lr = 0.0001
    return lr

learningrate_scheduler = LearningRateScheduler(get_learning_rate)

# Train model
print('Training model...')
hist = model.fit(datagen.flow(x_train, y_train, batch_size), epochs=epochs, verbose=1,
callbacks=[checkpointer, earlystopping, learningrate_scheduler], validation_data=(x_test, y_test))
print('Training complete...')

# View test results
#   Test accuracy
print('Running model on test data')
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]
print('Test complete.')
print('Test Accuracy: %.4f%%' % accuracy)

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