import numpy as np
from matplotlib import pyplot as plt
import csv
from task2_diffusionmap import diffusion_map, plot_dmaps

def subtask3():
    trajectories, data = load_vadere_data("data_DMAP_PCA_vadere.txt")
    plot_vadere_paths(trajectories)

    lambda_l, phi_l = diffusion_map(data, l=10)
    plot_subtask3(phi_l)
    plot_dmaps(phi_l, "task3_subtask3_4")


def plot_subtask3(data):
    print(data.shape)

    fig = plt.figure()
    plt.title('Diffusion map')
    plt.plot(data[:, 0], data[:,1])
    plt.xlabel("$\phi_1$")
    plt.ylabel("$\phi_2$")
    plt.show()
    plt.pause(0.5)
    fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask3_3.png', dpi=100)


def plot_vadere_paths(data, n=2):
    layout=[5,3]
    fig, ax = plt.subplots(*layout)

    for idx in range(data.shape[0]):
        ax[idx%layout[0],idx%layout[1]].plot(data[idx,:,0],data[idx,:,1])
        if (idx % layout[0] == layout[0]-1 and idx % layout[1] == 0):
            ax[idx%layout[0],idx%layout[1]].set_ylabel("$y$")
            ax[idx%layout[0],idx%layout[1]].set_xlabel("$x$")
    plt.show()
    plt.pause(0.5)
    fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask3_1.png', dpi=100)

    fig = plt.figure()
    plt.title('Pedestrian trajectories')
    for idx in range(data.shape[0]):
        if idx>=n:
            break
        plt.plot(data[idx, :, 0], data[idx, :, 1])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()
    plt.pause(0.5)
    fig.set_size_inches(8, 8)
    fig.savefig('task2_subtask3_2.png', dpi=100)

def load_vadere_data(filepath):
    array = np.genfromtxt(filepath, delimiter=' ')
    print(array.shape)
    trajectories = []
    for i in range(array.shape[1]):
        if (i+1)%2 == 0:
            trajectories.append(array[:,i-1:i+1])
    return np.array(trajectories), array


if __name__ == "__main__":
    plt.ion()
    subtask3()
    plt.ioff()
    plt.show()
