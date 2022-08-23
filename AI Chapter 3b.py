from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Load MNIST data from dataset
mnist_data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# Ensure samples are working correctly
print("The MNIST database has a training set of %d examples." % len(train_labels))
print("The MNIST database has a test set of %d examples." % len(test_labels))

# Show first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# View an image in more detail
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(train_images[0],ax)

# Preprocess so values are under 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Ensure samples are working correctly
print(train_images.shape)
print(test_images.shape)

num_classes = 10

# Print training labels
print('Integer valued labels:')
print(train_labels[:10])

# Convert vectors to binary matrices
train_labels=to_categorical(train_labels, num_classes)
test_labels=to_categorical(test_labels, num_classes)

# Print revised one-hot training labels
print('One-hot labels')
print(train_labels[:10])

# Reshape data to fit CNN
img_rows, img_cols = 28, 28
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print('input_shape: ', input_shape)
print('x_train shape:', train_images.shape)

# Build the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), strides=1, padding='same',
    activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer='rmsprop', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
hist = model.fit(train_images, train_labels, batch_size=32, epochs=12,
          validation_data=(test_images, test_labels), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

# Load model with best classification accuracy for this set
model.load_weights('model.weights.best.hdf5')
(test_loss, test_accuracy) = model.evaluate(test_images, test_labels, verbose=2)