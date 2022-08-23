from warnings import filters
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Dense, Input, Reshape, UpSampling2D, ReLU, Activation
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tensorflow.python import distribute
from tensorflow.python.keras import optimizers
from tensorflow.python.ops.gen_array_ops import pad

# File location
file_path = f'D:\\Keras\\Logs\\Ch8Project\\{time.strftime("%Y-%m-%d-%H%M%S")}\\'
os.makedirs(file_path, exist_ok=True)
latent_dim = 100
channels = 1
img_shape=(28,28,1)

# Defines the model for our fake image generator
def build_generator():
    generator = Sequential()

    #   FC1
    generator.add(Dense(units=128*7*7, activation='relu', input_dim=latent_dim))
    generator.add(Reshape((7,7,128)))

    #   UP2
    generator.add(UpSampling2D(size=(2,2)))

    #   Conv3
    generator.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(ReLU())

    #   UP4
    generator.add(UpSampling2D(size=(2,2)))

    #   Conv5
    generator.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(ReLU())

    #   Conv6
    generator.add(Conv2D(filters=channels, kernel_size=3, padding='same'))
    generator.add(Activation("tanh"))

    #   Print model summary
    generator.summary()

    #   Set input
    noise = Input(shape=(latent_dim,))

    #   Set output (fake image)
    img = generator(noise)

    return Model(inputs=noise, outputs=img)

# Defines the model for our discriminator
def build_discriminator():
    discriminator = Sequential()

    #   Conv1
    discriminator.add(Conv2D(filters=32, kernel_size=3, strides=2, 
    input_shape=img_shape, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    #   Conv2
    discriminator.add(Conv2D(filters=64,kernel_size=3, strides=2, padding='same'))
    discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    #   Conv3
    discriminator.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=(0.8)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    #   Conv4
    discriminator.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    #   FC5
    discriminator.add(Flatten())
    discriminator.add(Dense(units=1, activation='sigmoid'))

    #   Print model summary
    discriminator.summary()

    #   Define the input image
    img = Input(shape=img_shape)

    #   Define the ouput (probability)
    probability = discriminator(img)

    return Model(inputs=img, outputs=probability)

# Defines our function to visualize the image
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
    plt.show()

# Generate random images and plot these on the screen
# We will call this function while training so we can visually determine progress
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(28,28)):
    # Create random noise to feed into the generator
    noise = np.random.normal(0, 1, size=[examples, latent_dim])

    # Feed noise into the generator to predict new images
    # Resize these by removing the last dimension that is equal to 1
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    # Plot images to screen
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()

    # Save to file
    plt.savefig(f'{file_path}\gan_generated_image_epoch_%d.png' % epoch)

# Fetch data
print('Fetching data...')
(x_train, _), (_, _) = fashion_mnist.load_data()

# Prepare data
print('Preparing data...')

#   Normalize data
mean = np.mean(x_train, axis=(0,1, 2))
std = np.std(x_train, axis=(0,1, 2))
x_train = (x_train-mean)/(std+1e-7)
x_train = (x_train-mean)/(std+1e-7)
print(f'x_train shape: {x_train.shape}')

#   Visualize sample training data image
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
visualize_input(x_train[3343], ax)

# Create model
print('Preparing model...')
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
loss = BinaryCrossentropy()

#   Build the discriminator
discriminator = build_discriminator()
discriminator.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

#   Build the generator
generator = build_generator()

z = Input(shape=(latent_dim,))
img = generator(z)

discriminator.trainable = False
valid = discriminator(img)

#   Build the combined model
combined = Model(inputs = z, outputs = valid)
combined.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

def train(epochs, batch_size=128, save_interval=50):

    # Our ground truth ones and zeros labels
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator
        #   Selects a random group of images from our training group
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        #   Create some random noise and feed this data into the image generator
        #   Create up to 100 units of noise
        #   We will use this to generate fake images that can be fed into the discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        #   Calculate training loss
        #   We will do this twice, once for the real images and once for the fake images
        #   Finally, we will combine these together and calculate the result
        #discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the combined network (generator)
        #discriminator.trainable = False
        g_loss = combined.train_on_batch(noise, valid)

        # Print results
        print("%d [Discriminator loss: %f, acc.: %.2f%%]\n[Generator loss: %f, acc.: %.2f%%]" %
        (epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))

        # Plot generated images depending on the current save interval
        if epoch % save_interval == 0:
            plot_generated_images(epoch, generator)

#   Train the model
train(epochs=10000, batch_size=128, save_interval=500)