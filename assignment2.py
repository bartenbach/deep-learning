from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import random
from numpy.core.defchararray import startswith
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.np_utils import to_categorical


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


# vastly improved printing function sourced from
# https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48
# and modified
def plot_results(X: np.array, Y: np.array, Z: np.array, num: int) -> None:
    for i in range(num):
        plt.subplot(5, 4, i+1)
        plt.tight_layout()
        plt.imshow(X[i], cmap='gray')
        plt.title('Digit: {}, Pred: {}'.format(Y[i], Z[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


EPOCHS = 42

# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 1. print some images
plot_digits(x_train, y_train, 16)

# 2. Implement LeNet-5 with Keras functional API
# reshape data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# pad data
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))

# normalize data
x_train = tf.keras.utils.normalize(x_train, axis=-1, order=2)
x_test = tf.keras.utils.normalize(x_test, axis=-1, order=2)

# implement LeNet-5 using functional API
inputs = keras.Input(x_train[0].shape)
x = layers.Conv2D(filters=6,
                  kernel_size=5,
                  strides=1,
                  activation='relu')(inputs)
x = layers.MaxPool2D(pool_size=2, strides=2)(x)
x = layers.Conv2D(filters=16,
                  kernel_size=(5, 5),
                  strides=(1, 1),
                  activation='relu',
                  input_shape=(14, 14, 6))(x)
x = layers.MaxPool2D(pool_size=2, strides=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(units=120, activation=keras.activations.relu)(x)
x = layers.Dense(units=84, activation=keras.activations.relu)(x)
x = layers.Dense(units=10, activation=keras.activations.softmax)(x)

model = tf.keras.Model(inputs=inputs, outputs=x, name="mnist_model")
model.summary()

# this is neat
# keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)

# compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics='accuracy'
)

# 3. Train model and plot results
history = model.fit(x_train,
                    y_train,
                    validation_split=.3,
                    epochs=EPOCHS,
                    steps_per_epoch=10,
                    shuffle=True)


test_scores = model.evaluate(x_test, y_test, verbose=2)

# Plot error for training and validation
plt.title('Training error')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'], loc='upper right')
plt.show()

plt.title('Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train accuracy', 'val accuracy'], loc='upper right')
plt.show()

# 4. Conduct noise experiments on model
plt.title('Accuracy with noise at each layer')
plt.ylabel('Accuracy')
plt.xlabel('Noise')

# metrics storage
layer_stats = {}

# iterate through layers
for i in range(len(model.layers)):
    layer_name = 'layer-' + str(i)
    layer = model.layers[i]
    if (layer.name == "flatten" or layer.name.startswith("max_pooling2d")
            or layer.name.startswith("input")):
        continue
    layer_stats[layer_name] = []
    for j in np.arange(0.0, 2.25, 0.25):
        for weight in layer.trainable_variables:
            noise = np.random.normal(loc=0.0, scale=j, size=weight.shape)
            weight.assign_add(noise)
        test_scores = model.evaluate(x_test, y_test, verbose=2)
        layer_stats[layer_name].append(test_scores[1])
        for weight in layer.trainable_variables:
            noise = np.random.normal(loc=0.0, scale=j, size=weight.shape)
            weight.assign_sub(noise)

for key in layer_stats.keys():
    plt.plot([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
             layer_stats[key], label=key)
plt.legend(layer_stats.keys(), loc='upper right')
plt.show()

# 5. Conduct trainability experiments on model
plt.title('Accuracy with training one layer and randomizing the others')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

# iterate through layers
for i in range(len(model.layers)):
    layer_stats = {}
    # layer selected.  randomize other layers...
    fixed_layer = 'layer-' + str(i)
    layer_stats[fixed_layer] = []
    for j in range(len(model.layers)):
        inner_layer = 'layer-' + str(j)
        layer = model.layers[j]
        if j == i:
            layer.trainable = True
        else:
            if random.randint(0, 2) == 0:
                layer.trainable = True
            else:
                layer.trainable = False

    # layers randomized.  ready for model simulation.
    history = model.fit(x_train,
                        y_train,
                        validation_split=.3,
                        epochs=EPOCHS,
                        steps_per_epoch=10,
                        shuffle=True)

    test_scores = model.evaluate(x_test, y_test, verbose=2)

    # reset trainable attribute for next run
    for k in range(len(model.layers)):
        model.layers[k].trainable = True

    # plot results
    plt.legend(layer_stats.keys(), loc='upper right')
    plt.plot(history.history['accuracy'])
plt.show()
