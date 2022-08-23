import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load MNIST data from dataset
mnist_data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# Ensure samples are working correctly
print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)

# Preprocess so values are under 1
train_images = train_images / 255.0
test_images = test_images / 255.0

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

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()

# Compile the model
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="rmsprop", metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=20)

(test_loss, test_accuracy) = model.evaluate(test_images, test_labels,verbose=2)

# Make predictions
predictions = model.predict(test_images)
print("\nProblem 1:", predictions[0])

# Examine predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} ({:2.0f}%) ({})".format(predicted_label,
                                    100*np.max(predictions_array),
                                    [true_label],
                                    color=color)
    )

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()