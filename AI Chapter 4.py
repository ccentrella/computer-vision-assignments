from tensorflow import keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization

model = Sequential()

# Early Stopping function
keras.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)

# Add first layer and include L2 Regularization
model.add(Dense(units=16, kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='relu'))

# Add batch normalization
model.add(BatchNormalization())

# Add dropout layer AFTER normalization
model.add(Dropout())

# Add final output layer
model.add(Dense(5,'softmax'))

# Compile model
model.compile('adam','categorical-crossentropy',['accuracy','recall'])

# Data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
datagen.fit(training_set)