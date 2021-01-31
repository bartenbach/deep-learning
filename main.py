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
def get_k_fold_sets(x: np.array, k: int) -> Tuple[List[np.array], np.array]:
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
def get_bootstrap_sets(input: np.array, k: int, size: int) -> List[np.array]:
    sets = []
    for i in range(0, k):
        selection_locations = np.random.randint(low=0, high=len(input) - 1, size=size)
        selections = np.empty([50, 32, 32, 3])
        for j in range(0, len(selection_locations)):
            selections[j] = input[selection_locations[j]]
        sets.append(selections)
    return sets


# hold out validation sets
print("Partitioning sets using hold out validation method...")
(train, validate) = get_hold_out_sets(normalized_x_train)
print("Size of training array: " + str(len(train)))
print("Size of validation array: " + str(len(validate)) + "\n")

# k-fold validation using cross validation
# number of sets to use for k-fold
k = 5
print("Creating sets using k-fold cross validation with " + str(k) + " sets...")
(train, validate) = get_k_fold_sets(normalized_x_train, k)
print("training sets: " + str(len(train)))
for i in train:
    print("Size of training data: " + str(len(i)))
print("items in validation set: " + str(len(validate)) + "\n")

# creates 'k' sets of 'set_size' random elements
print("Creating sets using bootstrap validation method...")
k = 5
set_size = 50
print("Creating " + str(k) + " sets of size " + str(set_size))
sets = get_bootstrap_sets(normalized_x_train, k, set_size)
print("created training sets: " + str(len(sets)))
for i in sets:
    print("Size of training data: " + str(len(i)))
