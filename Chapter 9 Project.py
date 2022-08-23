from keras.preprocessing.image import load_img
from numpy.lib.function_base import disp
from tensorflow import keras
from tensorflow.keras.applications import VGG19, vgg19
import numpy as np
import tensorflow as tf

# Fetch data
print('Fetching data...')
base_image_path = keras.utils.get_file('paris.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
style_reference_image_path = keras.utils.get_file('starry_night.jpg', 'https://i.imgur.com/9ooB60I.jpg')
result_prefix = 'paris_generated'

# Display images
print('Loading images...')
import cv2
base_image = cv2.imread(base_image_path)
style_image = cv2.imread(style_reference_image_path)
cv2.imshow('Base Image', base_image)
cv2.waitKey(3000)
cv2.imshow('Style Image', style_image)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# Prepare data
print('Preprocessing data...')

#   Weights of different loss components
total_variation_weight = 1e-6
style_weight=1e-6
content_weight=2.5e-8

#   Dimensions of generated picture
width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# Open the specified image and reformat to appropriate tensors
def preprocess_image(image_path):
    # Load image
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
  
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

# Convert tensor to valid image
def deprocess_image(x):
    # Reshape image
    x = x.reshape((img_nrows, img_ncols, 3))

    # Normalize image
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Convert from BGR to RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Calculate gram matrix of image tensor (feature-wise outer product)
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

# Style loss is designed to maintain style of referenced style image
# in generated image. It is based on the gram matrices of feature maps
# from the style reference image and generated image
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size **2))

# Auxiliary loss function to main content of base image in generated image
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

# Calculate total variation loss
def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

# Create model
print('Creating model...')

#   Required variables
#   Layers for style loss
style_layer_names = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]

#   Content layer
content_layer_name = 'block5_conv2'

model = vgg19.VGG19(weights='imagenet', include_top=False)

# Get symbolic outputs of each 'key' layer
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Create model that returns features for each layer
feature_extractor = keras.Model(inputs = model.inputs, outputs = outputs_dict)

# Computes the loss
# Includes the sum of three losses, the content loss, style loss, and total variation loss
def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )

    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)

    return loss

# Computes the loss and gradients
@tf.function
def compute_loss_and_gradients(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

# Begin training
print('Beginning training...')
optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
))
iterations = 4000

for i in range (1, iterations + 1):
    loss, grads = compute_loss_and_gradients(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print('Iteration %d: loss=%.2f' % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        keras.preprocessing.image.save_img(fname, img)

print('Training complete.')

# Show generated image to user
print('Loading image...')
generated_image = cv2.imread(result_prefix + "_at_iteration_4000.png")
cv2.imshow('Generated Image', generated_image)