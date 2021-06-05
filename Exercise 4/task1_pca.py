import numpy as np


def pca(data):
    # Center matrix
    data_mean = np.mean(data, axis=0)
    data = data - data_mean

    # Decompose into singular vectors
    u, s, vh = np.linalg.svd(data)

    # Calculate principal component cooridanates
    pc_coordinates = np.dot(data, np.transpose(vh))
    return pc_coordinates, u, s, vh
    