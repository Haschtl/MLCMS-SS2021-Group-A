import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from scipy.spatial.distance import cdist
# from scipy.signal import find_peaks
from helper import InteractiveMode, show_and_save, load_data


def pca(data):
    """
    Apply Principal Component Analysis to data and return data in 
    principal component coordinates.

    Parameters
    ----------
    data : (N, M) array-like
        Data to be analyzed

    Returns
    -------
    pc_coordinates : (N, M) array-like
    """
    # Center matrix
    data_mean = np.mean(data, axis=0)
    data = data - data_mean

    # Decompose into singular vectors
    _, _, vh = np.linalg.svd(data)

    # Calculate principal component cooridanates
    pc_coordinates = np.dot(data, np.transpose(vh))
    return pc_coordinates

def format_func(value, tick_number):
    """
    Format velocity plots to have increments of pi in arclenght axis.
    """
    N = int(np.round(2 * value / 1000))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

def plot_data(x, y, name="", labels=["x", "y"], title=""):
    """
    Plot and save 2-D data.

    Parameters
    ----------
    x : (N, 1) array-like
        x coordinates of the data.
    y : (N, 1) array-like
        y coordinates of the data.
    name : string, optional
        Filename addition. Default: ""
    labels : list, optional
        Labels of axes. Default: ["x", "y"]
    title : string, optional
        Title of the plot. Default: ""
    """
    plt.figure()
    plt.plot(x, y, label=title)
    plt.title(title)
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    show_and_save("task5_{}".format(name))

def scatter_3d_data(x, y, z, name="", labels=["x", "y", "z"], title="", c=None, cmap=None, norm=None):
    """
    Scatter and save 3-D data.

    Parameters
    ----------
    x : (N, 1) array-like
        x coordinates of the data.
    y : (N, 1) array-like
        y coordinates of the data.
    z : (N, 1) array-like
        z coordinates of the data.
    name : string, optional
        Filename addition. Default: ""
    labels : list, optional
        Labels of axes. Default: ["x", "y", "z"]
    title : string, optional
        Title of the plot. Default: ""
    """
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter the values
    sc = ax.scatter(x, y, z, s=1, c=c, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    cb = plt.colorbar(sc, pad=0.1)
    cb.set_ticks([0,20,40,60,80,100,120,140,160,180,200])
    show_and_save("task5_{}".format(name))

def part1(data):
    """
    Takes three columns of the data and creates windows of size 
    (3, 351) at each time step. Then flattens these windows and applies
    PCA to represent data in principal component coordinates. Saves
    resulting coordinates to a .npy file.

    Parameters
    ----------
    data : (N, M) array-like
        Data read from "MI_timesteps.txt" file.
    """
    # pick columns 2,3,4
    first_3_meas_area = data[:,1:4]
    # create windows of size 351*3, flattened into vectors of size 1053
    windows = []
    for i in range(len(first_3_meas_area)-351):
        windows.append(first_3_meas_area[i:i+351].flatten())
    windows = np.array(windows)
    # NOTE: This operation takes long, so the results are stored in a npy file for future use.
    pc_coordinates = pca(windows)
    np.save('pc_coordinates.npy', pc_coordinates)

def part2(data):
    """
    Creates 9 scatter plots of the data in first three principal component
    coordinates. Each plot is colored according to values in corresponding
    observation column from the original data file.

    Parameters
    ----------
    data : (N, M) array-like
        Data read from "MI_timesteps.txt" file.
    """
    # load data in principal component coordinates and use only first three components (2d+1=3)
    pc_coordinates = np.load('pc_coordinates.npy')
    pc_coordinates = pc_coordinates[:,:3]
    # set colors for values in original file
    color_values = data[:-351,1:]
    # color the points on principal components' scatter plot according to each 9 observation column in original file
    for i in range(color_values.shape[1]):
        scatter_3d_data(pc_coordinates[:,0], pc_coordinates[:,1], pc_coordinates[:,2], 
                        c=color_values[:,i], cmap='rainbow', norm=Normalize(vmin=0, vmax=200), 
                        name="2_{}".format(i), title="Observations from area {} in embedding space.".format(i+1))

def part3(dt=1):
    """
    Create a velocity-arclength plot from the curve in in principal 
    component coordinates.

    Parameters
    ----------
    dt : int, optional
        Time step. Default: 1
    """
    # load data in principal component coordinates and use only first three components (2d+1=3)
    pc_coordinates = np.load('pc_coordinates.npy')
    pc_coordinates = pc_coordinates[:,:3]

    # Shift data in principal component coordinates by 1
    shifted_coords = np.roll(pc_coordinates, -dt, axis=0)
    # substracting data from its shifted version gives us velocity in each time step
    velocity = np.diag(cdist(pc_coordinates, shifted_coords))
    # plot velocity against time
    plot_data(np.arange(0, velocity.shape[0]-1,1), velocity[:-1], name="3", labels=["Arc length", "velocity"])
    # the velocity data is periodic and repeats in roughly every 2000 time steps
    one_period = velocity[:2000]
    # plot a single period
    plot_data(np.arange(0, one_period.shape[0],1), one_period, name="3_2", labels=["Arc length", "velocity"])
    plt.locator_params(axis='x', nbins=4)

    # arclengths = [np.sum(one_period[:i]) for i in range(one_period.shape[0])]
    # plot_data(np.arange(0, one_period.shape[0],1), arclengths, name="3_3", labels=["Time", "Arc length"])


if __name__ == "__main__":
    mi_timesteps = load_data("MI_timesteps.txt")
    # remove header and burn-in period of first 1000 time steps
    mi_timesteps = mi_timesteps[1001:]
    with InteractiveMode():
        part1(mi_timesteps)
        part2(mi_timesteps)
        part3(dt=1)
