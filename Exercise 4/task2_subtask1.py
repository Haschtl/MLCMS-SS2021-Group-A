import numpy as np
from matplotlib import pyplot as plt
from task2_diffusionmap import diffusion_map


def subtask1():
    """
    Execute subtask 1
    """
    part1_t_k, part1_dataset = part1dataset(n=1000)
    _, part1_eigenfunctions = diffusion_map(part1_dataset, l=5)
    plot_subtask1(part1_t_k, part1_dataset, part1_eigenfunctions)

    # fft = np.fft.fft(part1_dataset)
    # plt.plot(part1_t_k, fft)
    # # plt.plot(part1_t_k, fft[:,1])
    # # print(fft[:,0],fft[:,1])
    # plt.show()


def part1dataset(n=1000) -> np.ndarray:
    """
    Create the dataset for subtask 1
    """
    def x_k(k, n):
        t_k = (2*np.pi*k)/(n+1)
        return t_k, np.array([np.cos(t_k), np.sin(t_k)])
    X = np.zeros([n, 2])
    t_k = np.zeros([n])
    for idx, _ in enumerate(X):
        t_k[idx], X[idx] = x_k(idx+1, n)
    return t_k, X


def plot_subtask1(t_k, dataset, eigenfunctions):
    xlabel = "$t_k$"
    fig = plt.figure()
    plt.title('$x_k$ over $t_k$')
    plt.plot(t_k, dataset)
    # plt.plot(t_k, np.sum(dataset,1))
    plt.ylabel("$x_k$")
    plt.xlabel(xlabel)
    plt.show()
    plt.pause(0.5)
    # fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask1_1.png', dpi=100)

    fig = plt.figure()
    plt.title('$x_{k2}$ over $x_{k1}$')
    plt.plot(dataset[:, 0], dataset[:, 1])
    plt.ylabel("$x_{k1}$")
    plt.xlabel("$x_{k2}$")
    plt.show()
    plt.pause(0.5)
    # fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask1_2.png', dpi=100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(t_k, dataset[:, 0], dataset[:, 1])
    plt.show()
    plt.pause(0.5)
    plt.axis('tight')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$x_{k1}$")
    ax.set_zlabel("$x_{k2}$")
    plt.xticks([]), plt.yticks([])
    plt.title('3D')
    fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask1_3.png', dpi=100)

    fig, ax = plt.subplots(eigenfunctions.shape[1], 1)
    ax[0].set_title('Eigenvectors')
    for i in range(eigenfunctions.shape[1]):
        ax[i].scatter(t_k, eigenfunctions[:, i],  s=(72./fig.dpi)**2)
        ax[i].set_ylabel("$\phi_{}$".format(i+1))
    plt.show()
    plt.pause(0.5)
    plt.xlabel(xlabel)
    # fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask1_4.png', dpi=100)


if __name__ == "__main__":
    plt.ion()
    subtask1()
    plt.ioff()
    plt.show()
