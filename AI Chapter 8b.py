from PIL.ImageOps import pad
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, ReLU, Input
from tensorflow.python.keras.layers import noise

# Defines the model for our fake image generator
def generator_model():
    input_shape = 100
    generator = Sequential()

#   FC1
    generator.add(Dense(units=128*7*7, activation='relu', input_dim=input_shape))
    generator.add(Reshape((7,7,128)))

#   UP2
    generator.add(UpSampling2D(size=(2,2)))

#   Conv3
    generator.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation(ReLU()))

#   UP4
    generator.add(UpSampling2D(size=(2,2)))

#   Conv5
    generator.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation(ReLU()))

#   Conv6
    generator.add(Conv2D(filters=1, kernel_size=3, padding='same'))
    generator.add(Activation('tanh'))

#   Print model summary
    generator.summary()

#   Set input
    noise = Input(shape=(input_shape,))

#   Set output (fake image)
    fake_image = generator(noise)

    return Model(noise, fake_image)

generator_model()