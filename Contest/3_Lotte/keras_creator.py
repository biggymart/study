from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models
from tensorflow.keras import layers

conv_base = MobileNetV2(weights='imagenet',
     include_top=False,
     input_shape=(64, 64, 3))
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1000, activation='softmax'))

# model.summary()
conv_base.summary()