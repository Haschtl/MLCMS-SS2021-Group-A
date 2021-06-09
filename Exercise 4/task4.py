import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base

from task4_dataset import DataSet
from task3_model import vae



class FireEvacDataset:
    """
    Class to create dataset used by task3_model.py
    """

    def __init__(self, testset, trainset):
        self.train = Dataset(testset)
        self.test = Dataset(trainset)
        # self.train = tf.data.Dataset.from_tensor_slices(
        #     rescale(np.load(testset)))
        # self.test = tf.data.Dataset.from_tensor_slices(
        #     rescale(np.load(trainset)))


class Dataset:
    """
    Class to create dataset used by task3_model.py
    """

    def __init__(self, file):
        images = rescale(np.load(file))
        labels = np.zeros(images.shape[0])
        self.images = tf.data.Dataset.from_tensor_slices(images)
        self.labels = tf.data.Dataset.from_tensor_slices(labels)


def load_fireevac_dataset(testset, trainset):
    test_images = rescale(np.load(testset))
    test_labels = np.zeros(test_images.shape[0])
    train_images = rescale(np.load(trainset))
    train_labels = np.zeros(train_images.shape[0])
    train = DataSet(train_images, train_labels)
    validation = DataSet(np.array([]), np.array([]))
    test = DataSet(test_images, test_labels)
    return base.Datasets(train=train, validation=validation, test=test)
    # return FireEvacDataset(testset, trainset)


def rescale(array):
    print(array.shape)
    min_xy = np.min(array, 0)
    max_xy = np.max(array, 0)
    array = (array-min_xy)/max_xy  # scale to 0:1
    # array = (array*2)-1  # scale to -1:1
    return array


def visualize_dataset(dataset):
    fig = plt.figure()
    plt.scatter(dataset.test.images[:, 0],
                dataset.test.images[:, 1], s=(72./fig.dpi)**2)
    plt.scatter(dataset.train.images[:, 0],
                dataset.train.images[:, 1], s=(72./fig.dpi)**2)
    plt.title("FireEval dataset (train and test set)")
    plt.ylabel("$y$ (scaled to $[0;1]$)")
    plt.xlabel("$x$ (scaled to $[0;1]$)")
    plt.show()
    plt.savefig('task4_dataset.png', dpi=100)


def train(dataset):
    """
    Train the VAE with the MNIST dataset with the specified latent dimension
    """
    params = {
        "input_shape": [2, 1],
        "learning_rate": 0.0005,
        "latent_dim": 2,
        "encoder_units": 128,
        "decoder_units": 128,

        "epochs": 1200,
        "batch_size": 50,
        "iterations": 100,

        "test_after_epochs": [5, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,1100,1200],
        "num_samples": 1000,
    }

    model = vae(params)
    model.train(dataset)

    return model, dataset.train.images, dataset.train.labels, dataset.test.images, dataset.test.labels


def task4():
    dataset = load_fireevac_dataset(
        "FireEvac_test_set.npy", "FireEvac_train_set.npy")
    visualize_dataset(dataset)
    train(dataset)


if __name__ == "__main__":
    task4()
