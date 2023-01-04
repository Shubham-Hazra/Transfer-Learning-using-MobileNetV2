import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers.experimental.preprocessing import (RandomFlip,
                                                                RandomRotation)
from tensorflow.keras.preprocessing import image_dataset_from_directory

BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "dataset/"

train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                                  shuffle=False,
                                                  batch_size=TEST_BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  seed=42)

class_names = train_dataset.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmenter = tf.keras.Sequential(
    [RandomFlip("horizontal"), RandomRotation(0.4)])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)


def alpaca_model(IMG_SHAPE=IMG_SHAPE, data_augmentation=data_augmenter):
    inputs = tfl.Input(shape=IMG_SHAPE)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(0.2)(x)
    outputs = tfl.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model = alpaca_model()

learning_rate = 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset, validation_data=validation_dataset,
          epochs=20, verbose=1)

model.save('MobileNetV2.h5')

model = tf.keras.models.load_model('MobileNetV2.h5')

scores = model.evaluate(validation_dataset, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
