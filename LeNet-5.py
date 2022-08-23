from os import makedirs, mkdir
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Activation, Flatten, Dense, Softmax
from tensorflow.keras.activations import tanh, relu
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
import time

# Create file path for model
file_path = f'D:\\Keras\\Logs\\LeNet-5\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
makedirs(file_path, exist_ok=True)

# Load dataset
print('Fetching data...')
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Verify results
print('Verifying data...')
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Visualize images
print('Loading images...')
fig=plt.figure(figsize=(20,20))
for i in range(6):
    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(y_train[i])
plt.savefig(fname=f'{file_path}Sample.png')
plt.show()

# Show large image
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy = (y,x),
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if img[x][y] < thresh else 'black')

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
visualize_input(x_train[0], ax)
plt.show()

# Prepare data
#   Preprocess images
print('Preparing data...')
mean = np.mean(x_train)
std = np.std(x_train)
x_train=(x_train-mean)/(std+1e-7)
x_test =(x_test-mean)/(std+1e-7)

img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
print (f'X_train shape: {x_train.shape}')

#   Preprocess labels
num_classes = 10
y_train=to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define our model
model = Sequential()
print('Creating model...')

#   C1
model.add(Conv2D(filters=6, kernel_size=5, input_shape=(28, 28, 1), padding='same'))
model.add(Activation(relu))

#   S2
model.add(AveragePooling2D(pool_size=(2,2)))

#   C3
model.add(Conv2D(filters=16, kernel_size=5))
model.add(Activation(tanh))

#   S4
model.add(AveragePooling2D(pool_size=(2,2)))

#   C5
model.add(Conv2D(filters=120, kernel_size=5))
model.add(Activation(tanh))

#   Add required flatten layer
model.add(Flatten())

#   FC6
model.add(Dense(units=84))
model.add(Activation(tanh))

#   FC7
model.add(Dense(units=10))
model.add(Activation(Softmax()))

#   Summary
model.summary()

# Prepare the model
batch_size = 256
epochs=20
optimizer = Adam()
loss = CategoricalCrossentropy()
print('Compiling...')
model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

#   Set learning rate scheduled
def lr_schedule(epoch):
    if epoch <=2:
        return 0.0005
    elif epoch <= 5:
        return 0.0002
    elif epoch <= 8:
        return 0.0001
    elif epoch <= 12:
        return 0.00005
    else:
        return 0.00001
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

#   Set checkpointer
checkpointer = ModelCheckpoint(filepath=file_path+'{epoch:02d}-{val_loss:.2f}.h5',
     monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
print('Beginning training...')
hist = model.fit(x_train, y_train, batch_size, epochs,
    validation_data=(x_test, y_test), shuffle=True,
    verbose=1, callbacks=[lr_scheduler, checkpointer])
print('Training complete.')

# View results
#   Evaluate test accuracy
print('Running model on test data...')
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]
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
ax.set_ylabel('Accuracy')
plt.savefig(f'{file_path}Results-Loss.png')
plt.show()

print('Complete.')

