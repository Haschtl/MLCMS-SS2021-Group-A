import numpy as np
from matplotlib import pyplot as plt


def part1dataset(N = 1000) -> np.ndarray:
    def x_k(k, N):
        t_k = (2*np.pi*k)/(N+1)
        return t_k, np.array([np.cos(t_k), np.sin(t_k)])
    X = np.zeros([N, 2])
    T_k = np.zeros([N, 2])
    for idx, _ in enumerate(X):
        T_k[idx], X[idx] = x_k(idx+1, N)
    return T_k, X

def part2dataset(N = 1000) -> np.ndarray:
    def x_k(u, v):
        return np.array([u*np.cos(u),v,u*np.sin(u)])
    dataset = np.zeros([N,3])
    for idx, _ in enumerate(dataset):
        uv = np.random.rand(2)*10
        dataset[idx] = x_k(uv[0],uv[1])
    return dataset


def distance_matrix(matrix: np.ndarray): 
    distance = np.zeros([matrix.shape[0], matrix.shape[0], matrix.shape[1]])
    for i, row in enumerate(distance):
        for j, cell in enumerate(row):
            distance[i,j] = np.abs(matrix[i]-matrix[j])
    return distance
    # print(matrix)
    # m, n = np.meshgrid(matrix, matrix) 

    # return np.sum(np.abs(matrix[:, None] - matrix[None, :]), axis=-1)
    # return np.abs(m-n)


def diagonal_normalization_matrix(matrix: np.ndarray):
    # print(matrix)
    diag = np.zeros(matrix.shape)
    for i, row in enumerate(matrix):
        # print(row)
        rowsum = np.sum(row,0)
        # print(rowsum)
        diag[i,i] = rowsum
        # for j, cell in enumerate(row):
    return diag
    # print(matrix)
    # print(np.sum(matrix, (0, 1,2)))
    # return np.diag(np.sum(matrix, (1,2)))

def inv(matrix):
    return np.transpose(np.linalg.inv(np.transpose(matrix, (2, 0, 1))), (1,2,0))

def eig(matrix):
    u, v = np.linalg.eig(np.transpose(matrix, (2, 0, 1)))
    return np.transpose(u,(1,0)), np.transpose(v,(1,2,0))

def diffusion_map(matrix, L=5):
    # 1. Form a distance matrix D
    D = distance_matrix(matrix)
    # 2. Set epsilon to 5% of the diameter of the dataset
    # epsilon = 0.05*np.max(D, (0,1))
    epsilon = 0.05*np.max(D)
    print(epsilon)

    # 3. Form the kernel matrix W
    W = np.exp(-np.multiply(D,D)/epsilon)
    


    # 4. Form the diagonal normalization matrix P
    P = diagonal_normalization_matrix(W)

    # 5. Normalize W to form the kernel matrix K
    P_inv = inv(P)
    # P_inv = 1/P
    P_inv[P_inv == np.inf] = 0
    K = P_inv*W*P_inv
    # K = np.dot(np.dot(P_inv,W),P_inv)

    # 6. Form the diagonal normalization matrix Q
    Q = diagonal_normalization_matrix(K)
    
    # 7. Form the symmetric matrix T_hat
    # Q_pow = np.power(Q, -0.5)
    # Q_pow[Q_pow==np.inf] = 0
    Q_pow = np.power(Q, -0.5)
    Q_pow[Q_pow==np.inf] = 0
    T_hat = Q_pow*K*Q_pow
    # T_hat = np.dot(np.dot(Q_pow,K),Q_pow)
    

    print(T_hat)    
    # 8. Find the L + 1 largest eigenvalues a_l and associated eigenvectors v_l
    a_l, v_l = eig(T_hat)

    # a_l[L+1:]=0
    # v_l[L+1:]=0
    # 9. Compute the eigenvalues of T_hat^(1/epsilon) lambda_l
    lambda_l_square = np.power(a_l,1/epsilon)
    lambda_l = np.sqrt(lambda_l_square)

    # 10. Compute the eigenvectors phi_l of the matrix T
    phi_l = Q_pow*v_l

    # Return eigenvalues and eigenvectors
    return lambda_l, phi_l


def plot_swissroll(X, eigenvectors):
    fig = plt.figure()

    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.Spectral)

    ax.set_title("Original data")
    ax = fig.add_subplot(212)
    ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1], cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title('Projected data')
    plt.show()

def subtask2():
    X = part2dataset(N=1000)
    from pydiffmap import diffusion_map as dm
    
    neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}


    # lambda_l, phi_l = diffusion_map(X, n_eigen=10)
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=10, k=200, epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
    # fit to data and return the diffusion map.
    dmap = mydmap.fit_transform(X)
    print(dmap.shape)
    fig, ax = plt.subplots(dmap.shape[1], dmap.shape[1])
    for i in range(dmap.shape[1]):
        for j in range(dmap.shape[1]):
            if i == j:
                continue
            ax[i,j].scatter(dmap[:,i],dmap[:,j])

    plt.show()
    # print(lambda_l.shape)
    # print(phi_l.shape)
    # plot_swissroll(X, phi_l)

def subtask1():
    t_k, X = part1dataset(N=3)
    # X = np.transpose(X,(1,0))
    lambda_l, phi_l = diffusion_map(X)
    print(lambda_l)
    print(phi_l)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(phi_l[:, :, 0], phi_l[:, :,1])
    # plt.show()

if __name__ == "__main__":
    subtask2()
