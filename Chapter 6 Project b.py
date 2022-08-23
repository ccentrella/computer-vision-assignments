# Include necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from os import makedirs
import time

# Prepare location
file_path = f'D:\\Keras\\Logs\\VGGNet2\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
makedirs(file_path, exist_ok=True)

# Preprocess data
print('Preparing data...')
train_path = 'data2/train'
valid_path = 'data2/valid'
test_path = 'data2/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                          target_size=(224, 224),
                                                          batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                          target_size=(224, 224),
                                                          batch_size=30)

#   Load weights and apply global average pooling
base_model = vgg16.VGG16(weights='imagenet', include_top=False,
input_shape=(224, 224, 3))

#   Freeze all but last five layers
for layer in base_model.layers[:-5]:
    layer.trainable = False

#   Add classifier
last_output = base_model.output
x = Flatten()(last_output)
x = Dense(10, activation='softmax', name='softmax')(x)
new_model = Model(inputs=base_model.input, outputs=x)

#   Print model summary
new_model.summary()

# Compile model
print('Compiling...')
optimizer = Adam(learning_rate=0.0001)
loss = CategoricalCrossentropy()
new_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint(filepath=f'{file_path}Weights.h5',
    monitor='val_loss', save_best_only=True, verbose=1)

# Train model
print('Beginning training...')
history = new_model.fit(train_batches, steps_per_epoch=18,
validation_data=valid_batches, validation_steps=3,
epochs=20, verbose=1, callbacks=[model_checkpoint])
print('Training complete.')

# Evaluate model
print('Evaluating on test data...')
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
import numpy as np

#   Load the dataset of images at the specified location
def load_dataset(path):
  data = load_files(path)
  paths = np.array(data['filenames'])
  targets = to_categorical(np.array(data['target']))
  return paths, targets

test_files, test_targets = load_dataset('data2/test')

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

def path_to_tensor(img_path):
  img = image.load_img(img_path, target_size=(224,224))
  x = image.img_to_array(img)
  return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
  list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
  return np.vstack(list_of_tensors)

test_tensors = preprocess_input(paths_to_tensor(test_files))
print('Testing complete.')
print('\nTesting loss: {:.4f}\nTesting accuracy: {:.4f}'.format(
    *new_model.evaluate(test_tensors, test_targets)))

# Show confusion matrix
print('Loading confusion matrix...')
from sklearn.metrics import confusion_matrix
import numpy as np
cm_labels = ['0', '1', '2', '3', '4', '5','6','7','8','9']
cm = confusion_matrix(np.argmax(test_targets, axis=1),
np.argmax(new_model.predict(test_tensors), axis=1))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
indexes = np.arange(len(cm_labels))
for i in indexes:
  for j in indexes:
    plt.text(j, i, cm[i,j])
plt.xticks(indexes, cm_labels, rotation=90)
plt.xlabel('Predicted label')
plt.yticks(indexes, cm_labels)
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.show()

print('Finished.')