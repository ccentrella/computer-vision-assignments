from tensorflow.keras.applications.vgg16 import VGG16

# Fetch base model
base_model = VGG16(weights='imagenet', include_top=False,
    input_shape =(224,224,3))
base_model.summary()

# Freeze extraction layers
for layer in base_model.layers:
    layer.trainable = False

base_model.summary()

# Add classifier
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output

x = Flatten()(last_output)
x = Dense(2, activation='softmax', name='softmax')(x)

# Build model
new_model = Model(inputs=base_model.input, outputs=x)
new_model.summary()