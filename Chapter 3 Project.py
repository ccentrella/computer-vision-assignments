from typing import AsyncIterable
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.utils.data_utils import init_pool_generator
from tensorflow.python.util.tf_export import InvalidSymbolNameError

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

import numpy as np
import matplotlib.pyplot as plt

# Plot some of the data
fig = plt.figure(figsize=(20,5))
for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
plt.show()

# Rescale the images
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# Convert to one-hot encoding
from tensorflow.keras.utils import to_categorical
num_classes = (len(np.unique(y_train)))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Split data into training and validation
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Verify data has been split correctly
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# Our first layer
model.add(Conv2D(filters=32, kernel_size=2, padding='same',
    activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))

# Our second layer
model.add(Conv2D(filters=64, kernel_size=2, padding='same',
    activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Our third layer
model.add(Conv2D(filters=128, kernel_size=2, padding='same',
    activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Add dropout and flatten layers, before our regular fully-connected layers
model.add(Dropout(0.4))
model.add(Flatten())

# Add our first fully-connected layer
model.add(Dense(500, activation='relu',))
model.add(Dropout(0.6))

# Add our output layer
model.add(Dense(10, activation='softmax'))

# Print our model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['accuracy'])

# Automatically stop the model before overfitting
callback = (EarlyStopping(monitor='val_loss', min_delta=0, patience=15))

# Update weights when improved
from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1,
    save_best_only=True)

# Train the model
hist = model.fit(x_train, y_train, batch_size=512, epochs=1000,
    validation_data=(x_valid, y_valid), callbacks=[checkpointer, callback],
    verbose=1, shuffle=True)

# Load weights from model
model.load_weights('model.weights.best')

# Test model accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

# Show accuracy graphs
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='test')
plt.legend()
plt.show()