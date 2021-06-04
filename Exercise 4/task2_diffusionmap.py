import numpy as np
from matplotlib import pyplot as plt


def distance_matrix(matrix: np.ndarray):
    """
    Input: n*m
    Computes the distance matrix of a n*m matrix.
    n is the number of samples, m the dimension (e.g. 2 or 3)

    Output-shape: n*n
    """
    m, n = np.meshgrid(matrix, matrix)
    return np.sum(np.abs(matrix[:, None] - matrix[None, :]), axis=-1)


def diagonal_normalization_matrix(matrix: np.ndarray):
    """
    Input: 
        - matrix (n*m)
    Computes the diagonal normalization matrix:
    Diagonal elements represent the sum of their row.
    Returns: 
        - Diagonal matrix (n*m)
    """
    diag = np.zeros(matrix.shape)
    for i, row in enumerate(matrix):
        rowsum = np.sum(row, 0)
        diag[i, i] = rowsum
    return diag


def diffusion_map(matrix, l=5):
    """
    Input:
        - matrix (n*m)
        - l (int): The number of eigenvalue/eigenfunction pairs.

    Computes eigenvalues and eigenfunctions of the input matrix with the diffusion map algorithm.
    Returns: 
        - Eigenvalues    (l)
        - Eigenfunctions (n*l)
    """
    # 1. Form a distance matrix D
    D = distance_matrix(matrix)

    # 2. Set epsilon to 5% of the diameter of the dataset
    epsilon = 0.05*np.max(D)

    # 3. Form the kernel matrix W
    W = np.exp(-np.multiply(D, D)/epsilon)

    # 4. Form the diagonal normalization matrix P
    P = diagonal_normalization_matrix(W)

    # 5. Normalize W to form the kernel matrix K
    p_inv = np.linalg.inv(P)
    p_inv[p_inv == np.inf] = 0
    K = np.dot(np.dot(p_inv, W), p_inv)

    # 6. Form the diagonal normalization matrix Q
    Q = diagonal_normalization_matrix(K)

    # 7. Form the symmetric matrix T_hat
    q_pow = np.power(Q, -0.5)
    q_pow[q_pow == np.inf] = 0
    t_hat = np.dot(np.dot(q_pow, K), q_pow)

    # 8. Find the L + 1 largest eigenvalues a_l and associated eigenvectors v_l
    a_l, v_l = np.linalg.eig(t_hat)
    # Sort the eigenvalues
    idx = a_l.argsort()[::-1]
    a_l = a_l[idx]
    v_l = v_l[:, idx]

    # 9. Compute the eigenvalues of T_hat^(1/epsilon) lambda_l
    lambda_l_square = np.power(a_l, 1/epsilon)
    lambda_l = np.sqrt(lambda_l_square)

    # 10. Compute the eigenvectors phi_l of the matrix T
    phi_l = np.dot(q_pow, v_l)

    # Return eigenvalues and eigenvectors
    return lambda_l[:l], phi_l[:, :l]


def plot_dmaps(dmap, name):
    """
    Input:
        - dmap: list of Eigenfunctions
        - name: Name for plot filename
    Plots each eigenfunction over each other
    """
    fig, ax = plt.subplots(dmap.shape[1]-1, dmap.shape[1]-1)
    for i in range(dmap.shape[1]):
        for j in range(dmap.shape[1]):
            if i >= j:
                continue
            ax[i, j-1].scatter(dmap[:, i], dmap[:, j], s=(72./fig.dpi)**2)
            ax[i, j-1].set_xlabel("$\phi_{}$".format(i+1))
            ax[i, j-1].set_ylabel("$\phi_{}$".format(j+1))
    plt.show()
    plt.pause(0.5)
    fig.set_size_inches(8, 8)
    fig.savefig('{}.png'.format(name), dpi=100)
