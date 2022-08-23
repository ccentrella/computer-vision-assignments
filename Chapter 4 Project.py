# Import needed dependencies
from tensorflow.keras import callbacks
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers, optimizers
import numpy as np
from matplotlib import pyplot
import time

from tensorflow.python.keras.callbacks import EarlyStopping

# Fetch data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Print data shape
print('x_train =', x_train.shape)
print('x_valid =', x_valid.shape)
print('x_test =', x_test.shape)

# Prepare data for training
#   Normalize data
mean = np.mean(x_train, axis=(0,1,2,3))
std = np.std(x_train, axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_valid = (x_valid-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

#   One-hot encode labels
num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

#   Augment data
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)
datagen.fit(x_train)

# Build Model Architecture
#   Required-hyperparameters
base_hidden_units = 32
weight_decay = 1e-4
model = Sequential()

#   Conv1
model.add(Conv2D(base_hidden_units, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.L2(weight_decay),
        input_shape=(x_train.shape[1:])))
model.add(Activation('relu'))
model.add(BatchNormalization())

#   Conv2
model.add(Conv2D(base_hidden_units, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.L2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#   Pool1, Dropout1
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#   Conv3
model.add(Conv2D(base_hidden_units * 2, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.L2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#   Conv 4
model.add(Conv2D(base_hidden_units * 2, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.L2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#   Pool2, Dropout2
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#   Conv 5
model.add(Conv2D(base_hidden_units * 4, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.L2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#   Conv 6
model.add(Conv2D(base_hidden_units * 4, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.L2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#   Pool3, Dropout3
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

#   FC 1
model.add(Flatten())
model.add(Dense(100, activation='softmax'))

#   Print summary
model.summary()

#   Save model to file before operation
current_time = time.strftime("%Y-%m-%d-%H%M%S")
model.save(filepath=f'D:\\Keras\\Logs\\Ch4Project\\{current_time}.h5')

# Train the model
batch_size = 256
epochs = 125

checkpointer = ModelCheckpoint(filepath=f'D:\\Keras\\Logs\\Ch4Project\\{current_time}-'+'{epoch:02d}-{val_loss:.2f}.h5',
        verbose=1, save_best_only=True)
earlyStop = EarlyStopping(monitor='accuracy', patience=10)
optimizer = optimizers.Adam(learning_rate=0.0005, decay=1e-6)
#optimizer = optimizers.RMSprop(learning_rate=0.0003, decay=1e-6) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
        callbacks=[checkpointer, earlyStop], steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs, verbose=1, validation_data=(x_valid, y_valid))

# Save model to file before operation
model.save(filepath=f'D:\\Keras\\Logs\\Ch4Project\\{current_time} w/weights.h5')

# Evaluate the model
scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100, scores[0]))

# Show results
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.savefig(fname=f'D:\\Keras\\Logs\\Ch4Project\\{time.strftime("%Y-%m-%d-%H%M%S")}.png')
pyplot.show()