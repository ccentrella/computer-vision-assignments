from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Dense, Input

# Defines the model for our discriminator
def discriminator_model():
    input_shape = (28, 28, 1)
    discriminator = Sequential()

#   Conv1
    discriminator.add(Conv2D(filters=32, kernel_size=3, strides=2, 
    input_shape=input_shape, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

#   Conv2
    discriminator.add(Conv2D(filters=64,kernel_size=3, strides=2, padding='same'))
    discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

#   Conv3
    discriminator.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same'))
    discriminator.add(BatchNormalization(momentum=(0.8)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

#   FC4
    discriminator.add(Flatten())
    discriminator.add(Dense(units=1, activation='sigmoid'))

#   Print model summary
    discriminator.summary()

#   Define the input image
    img = Input(shape=input_shape)

#   Define the ouput (probability)
    probability = discriminator(img)

    return Model(img, probability)

discriminator_model()