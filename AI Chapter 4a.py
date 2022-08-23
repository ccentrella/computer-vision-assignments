from scipy.sparse.construct import random
from sklearn.datasets import make_blobs
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from tensorflow.python.keras.layers.normalization import BatchNormalization

# Generate simple dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2,
    cluster_std=2, random_state=2)
y = to_categorical(y)

# Split out dataset into training and test data
n_train = 800
train_X, test_X = X[:n_train], X[n_train:]
train_y, test_y = y[:n_train], y[n_train:]

# View our dataset structure
print(train_X.shape, test_X.shape)

# Develop model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(100, input_dim=2, activation='relu'))
model.add(Dense(200, input_dim=2, activation='relu'))

model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
    metrics=['accuracy'])

# View our model structure
model.summary()

# Train and evaluate our model
history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=200, verbose=1)
_, train_acc = model.evaluate(train_X, train_y)
_, test_acc = model.evaluate(test_X, test_y)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# Plot the model accuracy learning curves
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()