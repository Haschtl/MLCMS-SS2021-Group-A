input("Tensorflow v2.5 and tensorflow-probability is required! Press Enter to continue...")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from task3_model import train

# VAE MNIST Example from https://keras.io/examples/generative/vae/
# not from https://gist.github.com/danijar/1cb4d81fed37fd06ef60d08c1181f557 as it's for tensorflow@1.0

def download_mnist_dataset():
    return tf.keras.datasets.mnist.load_data()


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size,
            ] = digit

    fig = plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
    fig.set_size_inches(8, 8)
    fig.savefig('task3_latent_space.png', dpi=100)


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    fig = plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    fig.savefig('task3_label_cluster.png', dpi=100)


if __name__ == "__main__":
    plt.ion()
    params = {
        "epochs": 30,
        "batch_size": 128,
        "latent_dim": 2,
        "learning_rate": 0.001,
    }
    vae, x_train, y_train, x_test, y_test = train(params)
    plot_latent_space(vae, )
    plot_label_clusters(vae, x_train, y_train)
    plt.ioff()
    plt.show()
    # (x_train, y_train), (x_test, y_test) = download_mnist_dataset()
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # normalize between 0 and 1

    # split into training and testset

    
