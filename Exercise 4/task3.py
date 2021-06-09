import numpy as np
import matplotlib.pyplot as plt
from task3_model import train_mnist


def train_latent_dim(latent_dim):
    """
    Train the VAE with the MNIST dataset with the specified latent dimension
    """
    if latent_dim == 2:
        params = {
            "input_shape": [28, 28],
            "learning_rate": 0.001,
            "latent_dim": latent_dim,
            "encoder_units": 256,
            "decoder_units": 256,

            "epochs": 500,
            "batch_size": 128,
            "iterations": 60,

            "test_after_epochs": [1, 5, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "num_samples": 15,
        }
    else:
        params = {
            "input_shape": [28, 28],
            "learning_rate": 0.001,
            "latent_dim": latent_dim,
            "encoder_units": 256,
            "decoder_units": 128,

            "epochs": 1000,
            "batch_size": 128,
            "iterations": 60,

            "test_after_epochs": [1, 5, 50, 100, 200, 300, 400, 500,600,700,800,1000],
            "num_samples": 15,
        }
    vae, x_train, y_train, x_test, y_test = train_mnist(params)


if __name__ == "__main__":
    plt.ion()
    latent_dim = int(input("Enter the latent dimension for the VAE (e.g. 2 or 32): "))
    train_latent_dim(latent_dim)
    plt.ioff()
    plt.show()
