from typing import List, Tuple
from matplotlib.colors import Normalize
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Tuple of numpy arrays
# x_train, x_test uint8 arrays of RGB image data
# with shape (num_samples, 3, 32, 32)
# y_train, y_test unit8 arrays of category labels (integers in range 0-9)
# each with shape (num_samples, 1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# normalize dataset
normalized_x_train = tf.keras.utils.normalize(
    x_train, axis=-1, order=2
)


# prints out all images in a dataset
def print_all_images(input: np.array) -> None:
    for i in range(len(x_train)):
        plt.imshow(x_train[i])
        plt.axis("off")
        plt.show()


# splits training data into a 7:3 tuple for validation
def get_hold_out_sets(input: np.array) -> Tuple[np.array, np.array]:
    seventy_percent = int(len(input) * .7)
    return (input[:seventy_percent], input[seventy_percent:len(input)])


# splits training data using k-fold for validation
def get_k_fold_sets(x: np.array, k: int) -> Tuple[List[Tuple[np.array, np.array]], Tuple[np.array, np.array]]:
    set_size = int(len(normalized_x_train) / k)
    training = []
    validation = ()
    for i in range(0, k, 1):
        start = set_size * i
        training_bit = set_size * (i+1) - 1
        end = training_bit - 1
        training.append(x[start:end+1])
        validation += (x[training_bit],)
    return (training, validation)


# creates validation set using bootstrap method
def get_bootstrap_sets(input: np.array) -> List[Tuple[np.array, np.array]]:
    return None


# hold out validation sets
print("Partitioning sets using hold out validation method...")
(train, validate) = get_hold_out_sets(normalized_x_train)
print("Size of training array: " + str(len(train)))
print("Size of validation array: " + str(len(validate)) + "\n")

# k-fold validation sets
# TODO does not use cross validation - should be able to take size-1 elements
# in sets instead and then the last set is the validation set consisting of 1
# element from each set
# number of sets to use for k-fold
k = 5
print("Creating sets using k-fold cross validation with " + str(k) + " sets...")
(train, validate) = get_k_fold_sets(normalized_x_train, k)
print("training sets: " + str(len(train)))
print("items in validation set: " + str(len(validate)) + "\n")

# bootstrap sets
print("Creating sets using bootstrap validation method...")
sets = get_bootstrap_sets(normalized_x_train)
