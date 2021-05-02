import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# vastly improved printing function sourced from
# https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48
# and modified
def plot_digits(X: np.array, Y: np.array, num: int) -> None:
    for i in range(num):
        plt.subplot(5, 4, i+1)
        plt.tight_layout()
        plt.imshow(X[i], cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# -----------
# LeNet-5
# -----------
EPOCHS = 10

# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

plot_digits(x_train, y_train, 16)

# reshape data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# pad data
lenet_x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
lenet_x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))

# normalize data
lenet_x_train = tf.keras.utils.normalize(x_train, axis=-1, order=2)
lenet_x_test = tf.keras.utils.normalize(x_test, axis=-1, order=2)


def get_lenet(input):
    lenet_input = keras.Input(input[0].shape)
    lenet_output = layers.Conv2D(filters=6,
                                 kernel_size=5,
                                 strides=1,
                                 activation='relu')(lenet_input)
    lenet_output = layers.MaxPool2D(pool_size=2, strides=2)(lenet_output)
    lenet_output = layers.Conv2D(filters=16,
                                 kernel_size=(5, 5),
                                 strides=(1, 1),
                                 activation='relu',
                                 input_shape=(14, 14, 6))(lenet_output)
    lenet_output = layers.MaxPool2D(pool_size=2, strides=2)(lenet_output)
    lenet_output = layers.Flatten()(lenet_output)
    lenet_output = layers.Dense(
        units=120, activation=keras.activations.relu)(lenet_output)
    lenet_output = layers.Dense(
        units=84, activation=keras.activations.relu)(lenet_output)
    lenet_output = layers.Dense(
        units=10, activation=keras.activations.softmax)(lenet_output)
    lenet_model = tf.keras.Model(
        inputs=lenet_input, outputs=lenet_output, name="LeNet-5")
    return lenet_model


lenet_model = get_lenet(lenet_x_train)
lenet_model.summary()
lenet_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics='accuracy'
)

history = lenet_model.fit(lenet_x_train,
                          y_train,
                          validation_split=.3,
                          epochs=EPOCHS,
                          shuffle=True)

test_scores = lenet_model.evaluate(lenet_x_test, y_test, verbose=2)

# Plot error for training and validation
plt.title('LeNet-5 MNIST Training error')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'], loc='upper right')
plt.show()

# -------------
# MOBILENET
# -------------


IMAGE_SIZE = (224, 224)

# fresh data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
mobilenet_x_train = tf.keras.utils.normalize(x_train, axis=-1, order=2)
mobilenet_x_test = tf.keras.utils.normalize(x_test, axis=-1, order=2)


def resize_image(x):
    x = np.resize(x, IMAGE_SIZE)
    x = np.stack((x,)*3, axis=-1)
    return x.astype(np.float32)


def get_mobilenet(input):
    inputs = keras.Input(shape=input[0].shape)
    model = tf.keras.applications.MobileNet(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        input_shape=input[0].shape,
        pooling='avg'
    )
    for layer in model.layers:
        layer.trainable = False
    x = keras.layers.Dense(256, activation='relu')(model.output)
    x = keras.layers.Dense(10, activation='softmax')(x)
    mobileNet = tf.keras.Model(inputs, x, name='MobileNet')
    mobileNet.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=["accuracy"])
    return mobileNet


mobilenet_x_train = [resize_image(x) for x in x_train]
mobilenet_x_test = [resize_image(x) for x in x_test]
mobilenet_x_train = np.array(mobilenet_x_train)
mobilenet_x_test = np.array(mobilenet_x_test)
mobilenet_model = get_mobilenet(mobilenet_x_train)
mobilenet_model.summary()

mobilenet_initial_weights = mobilenet_model.get_weights()
history = mobilenet_model.fit(mobilenet_x_train,
                              y_train,
                              validation_split=.3,
                              epochs=EPOCHS,
                              shuffle=True)
mobilenet_optimized_weights = mobilenet_model.get_weights()

test_scores = mobilenet_model.evaluate(
    mobilenet_x_test, y_test, verbose=2)

# Plot error for training and validation
plt.title('MobileNet MNIST Training error')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'], loc='upper right')
plt.show()

lenet_model = get_lenet(lenet_x_train)
lenet_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
lenet_initial_weights = lenet_model.get_weights()
lenet_model.fit(
    lenet_x_train,
    y_train,
    epochs=EPOCHS,
    validation_split=.3
)
lenet_optimized_weights = lenet_model.get_weights()

alpha_history = []
lenet_loss_history = []
mobile_loss_history = []
alpha = 0.0
while alpha <= 2.0:
    # modify weights
    lenet_theta_weights = []
    mobile_theta_weights = []
    for i in range(len(lenet_model.get_weights())):
        theta = (1 - alpha) * \
            lenet_initial_weights[i] + (alpha * lenet_optimized_weights[i])
        lenet_theta_weights.append(theta)
    for i in range(len(mobilenet_model.get_weights())):
        theta = (1 - alpha) * \
            mobilenet_initial_weights[i] + \
                (alpha * mobilenet_optimized_weights[i])
        mobile_theta_weights.append(theta)
    lenet_model.set_weights(lenet_theta_weights)
    mobilenet_model.set_weights(mobile_theta_weights)
    alpha = alpha + 0.1

    # evaluate model at alpha
    alpha_history.append(alpha)
    lenet_test_scores = lenet_model.evaluate(lenet_x_test, y_test, verbose=2)
    mobile_test_scores = mobilenet_model.evaluate(
        mobilenet_x_test, y_test, verbose=2)
    lenet_loss_history.append(lenet_test_scores[0])
    mobile_loss_history.append(mobile_test_scores[0])
    print(test_scores)


f, ax = plt.subplots()
plt.plot(alpha_history, lenet_loss_history, '-', label='x')
plt.plot(alpha_history, mobile_loss_history, '-', label='x')
ax.legend(['LeNet', 'MobileNet'], loc=0)
ax.set_title('Comparison of LeNet-5 and MobileNet')
ax.set_xlabel('alpha')
ax.set_ylabel('loss')
plt.show()
