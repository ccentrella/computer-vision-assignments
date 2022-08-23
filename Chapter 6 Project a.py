# Required references
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow .keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.python.keras.losses import CategoricalCrossentropy

# Preprocess data
print('Preparing data...')
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                         target_size=(224,224),
                                                         batch_size=10)

valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                         target_size=(224,224),
                                                         batch_size=30)

# Load pre-trained model
base_model = vgg16.VGG16 (weights='imagenet', include_top=False,
                          input_shape = (224,224,3))

for layer in base_model.layers:
  layer.trainable = False

# Create model
print('Creating model...')
last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output

x = Flatten()(last_output)
x = Dense(64, activation='relu', name='FC_2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', name='softmax')(x)

new_model = Model(inputs=base_model.input, outputs=x)
new_model.summary()

# Compile model
print('Compiling model...')

optimizer = Adam(lr=0.0001)
loss = CategoricalCrossentropy()

new_model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

# Train model
print('Training model...')
new_model.fit(train_batches, steps_per_epoch=4,
                        validation_data=valid_batches, validation_steps=2,
                        epochs=20, verbose=1)
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

test_files, test_targets = load_dataset(test_path)

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
print('Finished.')
