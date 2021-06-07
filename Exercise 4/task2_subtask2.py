import numpy as np
from matplotlib import pyplot as plt
from task2_diffusionmap import diffusion_map, plot_dmaps
from task1_pca import pca

def subtask2(check_with_external_library=False):
    """
    Execute subtask 2
    """
    swiss_roll = part2dataset(n=5000)
    swiss_roll_small = part2dataset(n=1000)

    plot_swissroll(swiss_roll, "swiss_roll")
    plot_swissroll(swiss_roll_small, "swiss_roll_small")

    if check_with_external_library:
        from pydiffmap import diffusion_map as dm
        neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
        mydmap = dm.DiffusionMap.from_sklearn(
            n_evecs=10, k=200, epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
        # fit to data and return the diffusion map.
        dmap = mydmap.fit_transform(swiss_roll)
        plot_dmaps(dmap, "task2_subtask2_eigenfunctions_external_library")
    else:
        lambda_l, phi_l = diffusion_map(swiss_roll, l=10)
        # plot_dmaps(phi_l, "task2_subtask2_eigenfunctions")
        plot_subtask2(phi_l, "eigenfunctions")

        lambda_l, phi_l = diffusion_map(swiss_roll_small, l=10)
        # plot_dmaps(phi_l, "task2_subtask2_eigenfunctions")
        plot_subtask2(phi_l, "eigenfunctions_n1000")

    pc_coordinates, u, s, vh = pca(swiss_roll)
    plot_subtask2(pc_coordinates, "pca_n5000")
    pc_coordinates, u, s, vh = pca(swiss_roll_small)
    plot_subtask2(pc_coordinates, "pca_n1000")


def part2dataset(n=1000) -> np.ndarray:
    """
    Create the dataset for subtask 2
    """
    def x_k(u, v):
        return np.array([u*np.cos(u), v, u*np.sin(u)])
    dataset = np.zeros([n, 3])
    for idx, _ in enumerate(dataset):
        uv = np.random.rand(2)*10
        dataset[idx] = x_k(uv[0], uv[1])
    return dataset


def swissroll_color(swiss_roll):
    """
    Calculate colors for swiss-roll depending on the distance from zero (dim 0 and 2)
    """
    R = np.sqrt(swiss_roll[:, 0]**2 + swiss_roll[:, 2]**2)
    R = R/np.amax(R)
    return R


def plot_swissroll(swiss_roll, name):
    """
    Plot swissroll in 3d
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(swiss_roll[:, 0], swiss_roll[:, 1],
               swiss_roll[:, 2], c=swissroll_color(swiss_roll))
    plt.show()
    plt.pause(0.5)
    plt.axis('tight')
    ax.set_xlabel("$x_{k1}$")
    ax.set_ylabel("$x_{k2}$")
    ax.set_zlabel("$x_{k3}$")
    plt.xticks([]), plt.yticks([])
    plt.title('Original data')
    fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask2_{}.png'.format(name), dpi=100)


def plot_subtask2(dmap, name):
    fig, ax = plt.subplots(dmap.shape[1]-1)
    for i in range(dmap.shape[1]):
        if i == 0:
            continue
        ax[i-1].scatter(dmap[:, 0], dmap[:, i], s=(72./fig.dpi)**2)
        ax[i-1].set_xlabel("$\phi_{}$".format(1))
        ax[i-1].set_ylabel("$\phi_{}$".format(i+1))
    plt.show()
    plt.pause(0.5)
    fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask2_{}.png'.format(name), dpi=100)


if __name__ == "__main__":
    plt.ion()
    subtask2()
    plt.ioff()
    plt.show()
