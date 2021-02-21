from typing import Generator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import Normalize
from tensorflow import keras
from tensorflow.keras import layers

# Tuple of numpy arrays
# x_train, x_test uint8 arrays of RGB image data
# with shape (num_samples, 3, 32, 32)
# y_train, y_test unit8 arrays of category labels (integers in range 0-9)
# each with shape (num_samples, 1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# vastly improved printing function sourced from
# https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48
# and modified
def plot_digits(X: np.array, Y: np.array, num: int) -> None:
    for i in range(num):
        plt.subplot(5, 4, i+1)
        plt.tight_layout()
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# prints out images in a dataset (old method, one by one)
def print_all_images(input: np.array, number: int) -> None:
    for i in range(number):
        plt.imshow(x_train[i])
        plt.axis("off")
        plt.show()


# 1. print some images
# plot_digits(x_train, y_train, 16)

# 2. Implement LeNet-5 with Keras functional API
# create validation set
x_validate, y_validate = x_train[55000:], y_train[55000:]

print("Image Shape: {}".format(x_train.shape), end='\n\n')
print("Training Set:   {} samples".format(len(x_train)))
print("Validation Set:   {} samples".format(len(x_validate)))
print("Test Set:       {} samples".format(len(x_test)))

# reshape data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# normalize data
x_train = tf.keras.utils.normalize(x_train, axis=-1, order=2)
x_test = tf.keras.utils.normalize(x_test, axis=-1, order=2)
print("Image Shape: {}".format(x_train.shape), end='\n\n')

# shape of data is (60,000, 28, 28) test set is 10,000 images
inputs = keras.Input(x_train.shape)
x = layers.Conv2D(filters=6,
                  kernel_size=5,
                  strides=1,
                  activation='relu',
                  input_shape=(32, 32, 1))(inputs)

x = layers.MaxPool2D(pool_size=2, strides=2)(x[0])
x = layers.Conv2D(filters=16,
                  kernel_size=(5, 5),
                  strides=(1, 1),
                  activation='relu',
                  input_shape=(14, 14, 6))(x)
x = layers.MaxPool2D(pool_size=2, strides=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(units=120, activation='relu')(x)
x = layers.Dense(units=84, activation='relu')(x)
x = layers.Dense(units=10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=x, name="mnist_model")
model.summary()
