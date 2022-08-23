from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ReLU, Softmax, Flatten, Dropout
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import GlorotNormal
from os import makedirs
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

# Create file path for model
file_path = f'D:\\Keras\\Logs\\VGGNet3\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
makedirs(file_path, exist_ok=True)

# Load dataset
print('Fetching data...')
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data(label_mode='fine')

# Verify results
print('Verifing results...')
assert x_train.shape == (50000,32,32,3)
assert x_test.shape == (10000,32,32,3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# Prepare data
#   Normalize images
print('Preparing images...')
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean)/(std + 1e-7)
x_test = (x_test - mean)/(std + 1e-7)

#   One-hot encode labels
num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Create model
print('Creating model')
model = Sequential(name='VGGNET')
base_filter_count = 64

#   C1
model.add(Conv2D(filters=base_filter_count, kernel_size=(3,3), padding='same',
    input_shape=(32,32,3), kernel_initializer='glorot_normal', kernel_regularizer=L2(0.0005)))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#   C3
model.add(Conv2D(filters=2*base_filter_count, kernel_size=(3,3), padding='same'))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2 )))

#   C5
model.add(Conv2D(filters=4*base_filter_count, kernel_size=(3,3), padding='same'))
model.add(ReLU())

#   C6
model.add(Conv2D(filters=4*base_filter_count, kernel_size=(3,3), padding='same'))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#   C8
model.add(Conv2D(filters=8*base_filter_count, kernel_size=(3,3), padding='same'))
model.add(ReLU())

#   C9
model.add(Conv2D(filters=8*base_filter_count, kernel_size=(3,3), padding='same'))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#   C11
model.add(Conv2D(filters=8*base_filter_count, kernel_size=(3,3), padding='same'))
model.add(ReLU())

#   C12
model.add(Conv2D(filters=8*base_filter_count, kernel_size=(3,3), padding='same'))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#   Flatten layer
model.add(Flatten())

#   FC14
model.add(Dense(units=4096, kernel_regularizer=L2(0.0005)))
model.add(ReLU())
model.add(Dropout(0.5))

#   FC15
model.add(Dense(units=4096, kernel_regularizer=L2(0.0005)))
model.add(ReLU())
model.add(Dropout(0.5))

#   FC16
model.add(Dense(units=100))
model.add(Softmax())

#   Print summary
model.summary()

# Configure data augmentation
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
epochs = 1000
initial_lr = 0.01
optimizer = SGD(learning_rate=initial_lr, momentum=0.9)
loss = CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12)
model_checkpoint = ModelCheckpoint(filepath=f'{file_path}Weights.h5',
    monitor='val_loss', save_best_only=True, verbose=1)

# Begin training
print('Beginning training...')
hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
    callbacks=[early_stopping, model_checkpoint, reduce_lr], validation_data=(x_test, y_test), verbose=1)
print('Training complete...')

# View test resuls
#   Test accuracy
print('Calculating test accuracy...')
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]
print('Test complete.')
print('Test accuracy: %.4f%%' % accuracy)

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